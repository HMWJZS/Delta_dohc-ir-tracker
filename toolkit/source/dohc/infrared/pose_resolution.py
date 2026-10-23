"""Causal selection of current-image planar poses; never publish held poses.

Continuity is a motion assumption, not an independent accuracy measurement.
Translation agreement can provide position without resolving orientation.
The online policy keeps both branches and publishes one current measurement.
"""
from itertools import combinations, permutations
import cv2
import numpy as np

from .config import CONFIG
from .profiling import timed


def rotation_distance_deg(first, second):
    a = cv2.Rodrigues(np.asarray(first['rvec'], dtype=float))[0]
    b = cv2.Rodrigues(np.asarray(second['rvec'], dtype=float))[0]
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T)-1)/2, -1, 1))))


class PoseResolver:
    def __init__(self):
        self.anchors = {}
        self.pending = {}
        self.positions = {}
        self.hypotheses = {}
        self.reliable = {}
        self.outputs = {}

    def discard(self, name):
        self.positions.pop(name, None)
        self.discard_orientation(name)

    def discard_orientation(self, name):
        self.anchors.pop(name, None)
        self.pending.pop(name, None)
        self.hypotheses.pop(name, None)
        self.reliable.pop(name, None)

    def retain(self, names, stamp):
        """Missing observations, expired source times and identity conflicts reset."""
        for state in (self.anchors, self.pending, self.positions, self.hypotheses, self.reliable):
            for name, old in list(state.items()):
                if name not in names or not 0 < stamp-old[0] <= CONFIG['pose_history_max_gap_seconds']:
                    self.discard(name)

    @staticmethod
    def translation_limit(dt):
        return min(CONFIG['pose_translation_step_limit_m'],
            CONFIG['pose_translation_slack_m'] + CONFIG['pose_translation_speed_limit_m_s']*dt)

    @staticmethod
    def rotation_limit(dt):
        return min(CONFIG['pose_rotation_step_limit_deg'],
            CONFIG['pose_rotation_slack_deg'] + CONFIG['pose_rotation_speed_limit_deg_s']*dt)

    @staticmethod
    def bounded(old, pose, stamp):
        dt = stamp-old[0]
        if not 0 < dt <= CONFIG['pose_history_max_gap_seconds']:
            return False
        rotation_limit = PoseResolver.rotation_limit(dt)
        translation_limit = PoseResolver.translation_limit(dt)
        translation = np.linalg.norm(np.asarray(pose['tvec_m'])-old[1]['tvec_m'])
        return rotation_distance_deg(old[1], pose) <= rotation_limit and translation <= translation_limit

    def resolve(self, detection, stamp, identity_confirmed):
        name = detection['pattern_id']
        old = self.positions.get(name)
        if old is not None and not 0 < stamp-old[0] <= CONFIG['pose_history_max_gap_seconds']:
            self.discard(name)
        # Position selection uses only the history preceding this source image.
        self.resolve_pose(detection, stamp, identity_confirmed)
        self.resolve_position(detection, stamp, identity_confirmed)

    def resolve_continuous(self, detection, stamp, identity_confirmed):
        """Seed with minimum RMS, then follow a bounded current-image branch."""
        name = detection['pattern_id']
        detection['pose_ambiguous_single_frame'] = detection['pose_ambiguous']
        evidence = dict(status='unresolved', reason='identity_unconfirmed',
            anchor_ready=False, output_supported=False, source_monotonic=stamp)
        detection['pose_resolution'] = evidence
        measurements = self.current_measurements(detection)
        anchor = self.anchors.get(name)
        if anchor is not None and not 0 < stamp-anchor[0] <= CONFIG['pose_history_max_gap_seconds']:
            self.discard(name)
            anchor = None
        if not identity_confirmed or not measurements:
            self.discard(name)
            if identity_confirmed:
                evidence['reason'] = 'insufficient_current_candidates'
            detection.pop('pose', None)
            self.resolve_position(detection, stamp, identity_confirmed)
            return
        if anchor is None:
            selected = min(measurements, key=lambda m: m['pose']['rms_px'])
            status = 'minimum_rms_initialization'
            step = None
        else:
            bounded = [m for m in measurements if self.bounded(anchor, m['pose'], stamp)]
            if not bounded:
                evidence.update(reason='motion_limit', anchor_ready=True)
                detection.pop('pose', None)
                self.resolve_position(detection, stamp, identity_confirmed)
                return
            selected = min(bounded, key=lambda m: (
                rotation_distance_deg(anchor[1], m['pose']), m['pose']['rms_px']))
            status = 'continuous_branch'
            step = rotation_distance_deg(anchor[1], selected['pose'])
        detection.update(pose=selected['pose'],
            refinement_applied=selected.get('refinement_applied', detection.get('refinement_applied', False)),
            robust_cost=selected.get('robust_cost', detection.get('robust_cost')),
            pose_selection=dict(policy='minimum_rms_continuous',
                stage=selected.get('candidate_stage', 'G' if detection.get('refinement_applied') else 'U'),
                source_monotonic=stamp,
                criterion='minimum_rms' if anchor is None else 'minimum_rotation_step_then_rms'))
        self.anchors[name] = (stamp, selected['pose'])
        evidence.update(status=status, reason='accepted_current_candidate', anchor_ready=True,
            output_supported=True, rotation_step_deg=step,
            u_candidate_index=selected['u_candidate_index'])
        self.resolve_position(detection, stamp, identity_confirmed)

    @staticmethod
    def temporal_prediction(history, stamp):
        recent = [(t, p) for t, p in history
                  if 0 < stamp-t <= CONFIG['pose_temporal_max_age_seconds']]
        if len(recent) < 2:
            return None
        (ta, a), (tb, b) = recent[-2:]
        if not ta < tb < stamp:
            return None
        ratio = (stamp-tb)/(tb-ta)
        pa, pb = np.asarray(a['tvec_m']), np.asarray(b['tvec_m'])
        ra = cv2.Rodrigues(np.asarray(a['rvec'], dtype=float))[0]
        rb = cv2.Rodrigues(np.asarray(b['rvec'], dtype=float))[0]
        step = cv2.Rodrigues(ra.T @ rb)[0].ravel()
        return pb+(pb-pa)*ratio, rb @ cv2.Rodrigues(step*ratio)[0]

    @staticmethod
    def temporal_cost(prediction, pose):
        if prediction is None:
            return 0.
        position, rotation = prediction
        r = cv2.Rodrigues(np.asarray(pose['rvec'], dtype=float))[0]
        angle = np.linalg.norm(cv2.Rodrigues(rotation.T @ r)[0])
        x = np.linalg.norm(np.asarray(pose['tvec_m'])-position)/CONFIG['pose_temporal_translation_scale_m']
        y = angle/np.deg2rad(CONFIG['pose_temporal_rotation_scale_deg'])
        return float(np.hypot(1., x)-1 + np.hypot(1., y)-1)

    @timed('dual_temporal_selection')
    def resolve_dual(self, detection, stamp, identity_confirmed, search_complete=True):
        """Rank current measurements while separating association from prediction."""
        name = detection['pattern_id']
        detection['pose_ambiguous'] |= len(detection.get('pose_candidates', [])) < 2
        detection['pose_ambiguous_single_frame'] = detection['pose_ambiguous']
        evidence = dict(status='unresolved', reason='identity_unconfirmed',
            anchor_ready=False, output_supported=False, orientation_supported=False,
            source_monotonic=stamp)
        detection['pose_resolution'] = evidence
        values = self.current_measurements(detection, all_branches=True)
        points = len(detection.get('points_raw', detection.get('led_ids', [])))
        values = [m for m in values if points and
                  np.isfinite(m.get('robust_cost', np.nan)) and m['robust_cost'] >= 0]
        previous = self.outputs.get(name)
        old = self.hypotheses.get(name)
        if not np.isfinite(stamp) or previous is not None and stamp <= previous[0]:
            values = []
            evidence['reason'] = 'invalid_source_time'
        if not identity_confirmed or not values:
            self.discard(name)
            detection.pop('pose', None)
            if identity_confirmed and evidence['reason'] != 'invalid_source_time':
                evidence['reason'] = 'insufficient_current_candidates'
            self.resolve_position(detection, stamp, identity_confirmed)
            return
        reset_reason = None
        if old is not None and stamp-old[0] > CONFIG['pose_history_max_gap_seconds']:
            self.discard(name)
            old = None
            reset_reason = 'association_expired'
        unary = lambda m: m['robust_cost']/(2*points)
        values = sorted(values, key=unary)[:2]
        matches, steps = {}, [None]*len(values)
        active, streak = None, 0
        if old is not None:
            dt = stamp-old[0]
            distances = [[(rotation_distance_deg(t['pose'], m['pose'])/self.rotation_limit(dt),
                float(np.linalg.norm(np.asarray(t['pose']['tvec_m'])-m['pose']['tvec_m']))/
                self.translation_limit(dt)) for m in values] for t in old[1]]
            count = min(len(old[1]), len(values))
            assignments = [list(zip(indices, order))
                for indices in combinations(range(len(old[1])), count)
                for order in permutations(range(len(values)), count)]
            pairs = min(assignments, key=lambda pairs: (
                sum(max(distances[i][j]) > 1 for i, j in pairs),
                sum(sum(distances[i][j]) for i, j in pairs)))
            matches = {j: i for i, j in pairs}
            # Preserve branch slots across changes in IPPE candidate count/order.
            order = sorted(range(len(values)), key=lambda j: matches.get(j, len(old[1])+j))
            values = [values[j] for j in order]
            matches = {j: matches[k] for j, k in enumerate(order) if k in matches}
            steps = [distances[matches[j]][order[j]] if j in matches else None
                     for j in range(len(values))]
            active = next((j for j, i in matches.items() if i == old[2]), None)
            streak = old[3]
        legal = [step is not None and max(step) <= 1 for step in steps]
        reliable_history = self.reliable.get(name, (stamp, []))[1]
        common = self.temporal_prediction(reliable_history, stamp)
        provisional = (self.temporal_prediction(old[1][old[2]]['history'], stamp)
                       if old is not None else None)
        prediction = common if common is not None else provisional
        basis = ('reliable_history' if common is not None else
                 'selected_branch_history' if provisional is not None else 'image_only')
        scores = [unary(m)+self.temporal_cost(prediction, m['pose']) for m in values]
        roots = sorted(detection.get('pose_candidates', []), key=lambda p: p['rms_px'])
        def single_support(m):
            return (not detection['pose_ambiguous_single_frame'] and len(roots) >= 2 and
                    m['u_candidate_index'] == 0 and roots[1]['rms_px']-roots[0]['rms_px'] >= .25)
        support_frames = [(old[1][matches[i]].get('single_support_frames', 0)+1
                           if old is not None and i in matches else 1) if single_support(m) else 0
                          for i, m in enumerate(values)]
        # The published branch also has to pass the gate, not just its own rival history.
        eligible = [i for i, m in enumerate(values)
                    if old is not None and previous is not None and
                    self.bounded(previous[:2], m['pose'], stamp)]
        status = 'dual_continuity'
        if old is None or not eligible:
            active = min(range(len(values)), key=lambda i: (scores[i], i))
            status = 'minimum_image_cost_initialization' if old is None else 'motion_reinitialized'
            reset_reason = reset_reason or ('cold_start' if previous is None else
                'history_reset' if old is None else 'no_output_continuous_candidate')
            streak = 0
        else:
            best = min(eligible, key=lambda i: (scores[i], i))
            if active not in eligible:
                active, streak, status = best, 0, 'motion_branch_switch'
            else:
                supported_rivals = [i for i, m in enumerate(values)
                                    if i != active and single_support(m) and i not in eligible]
                challenger = supported_rivals[0] if supported_rivals else best
                if supported_rivals:
                    streak = 0
                    if support_frames[challenger] >= CONFIG['pose_switch_frames']:
                        active, status = challenger, 'single_frame_branch_correction'
                        reset_reason = 'persistent_single_frame_evidence'
                elif challenger != active and scores[active]-scores[challenger] > CONFIG['pose_switch_margin']:
                    streak += 1
                    if streak >= CONFIG['pose_switch_frames']:
                        active, streak = challenger, 0
                        status = 'score_branch_switch'
                else:
                    streak = 0
        selected = values[active]
        step = rotation_distance_deg(previous[1], selected['pose']) if previous is not None else None
        segment = (previous[2]+int(reset_reason is not None)) if previous else 1
        tracks = []
        for i, m in enumerate(values):
            history = old[1][matches[i]]['history'] if old is not None and i in matches else []
            history = [(t, p) for t, p in history
                       if 0 < stamp-t <= CONFIG['pose_history_max_gap_seconds']] if legal[i] else []
            violations = (old[1][matches[i]]['violations']+1
                          if old is not None and i in matches and not legal[i] else 0)
            tracks.append(dict(pose=m['pose'], history=(history+[(stamp, m['pose'])])[-2:],
                               violations=violations, single_support_frames=support_frames[i]))
        single = bool(single_support(selected))
        motion = (len(tracks) == 2 and legal[active] and reset_reason is None and
                  tracks[1-active]['violations'] >= CONFIG['pose_switch_frames'])
        ordered_scores = sorted(scores[i] for i in eligible)
        margin = ordered_scores[1]-ordered_scores[0] if len(ordered_scores) == 2 else None
        anchored = (common is not None and reset_reason is None and active in eligible and
                    margin is not None and margin > CONFIG['pose_switch_margin'] and
                    scores[active] == ordered_scores[0])
        supported = bool(single or motion or anchored)
        detection['pose_ambiguous'] |= not single
        detection.update(pose=selected['pose'],
            refinement_applied=selected.get('refinement_applied', False),
            robust_cost=selected['robust_cost'],
            pose_selection=dict(policy=CONFIG['pose_output_policy'],
                stage=selected.get('candidate_stage', 'U'), source_monotonic=stamp,
                criterion='normalized_image_and_output_continuity',
                temporal_model='baseline_v9_causal_output_continuity', hypothesis_index=active))
        if reset_reason is not None:
            # A discontinuity must not transport reliable or provisional direction evidence.
            for i, track in enumerate(tracks):
                track['history'] = [(stamp, track['pose'])]
                track['violations'] = 0
                track['single_support_frames'] = int(single_support(values[i]))
            reliable_history = []
        self.hypotheses[name] = (stamp, tracks, active, streak)
        self.anchors[name] = (stamp, selected['pose'])
        self.outputs[name] = (stamp, selected['pose'], segment)
        reliable = [(t, p) for t, p in reliable_history
                    if 0 < stamp-t <= CONFIG['pose_temporal_max_age_seconds']]
        if supported and search_complete:
            reliable = (reliable+[(stamp, selected['pose'])])[-2:]
        self.reliable[name] = (stamp, reliable)
        evidence.update(status=status, reason='accepted_current_candidate', anchor_ready=True,
            output_supported=True, orientation_supported=supported,
            orientation_basis=('single_frame' if single else 'motion_assumption' if motion
                else 'reliable_history_motion_assumption' if anchored else 'unresolved'),
            candidate_scores=scores, score_margin=margin, history_basis=[basis]*len(values),
            reliable_history_frames=len(reliable), search_complete=bool(search_complete),
            challenger_frames=streak, candidate_motion_violations=[t['violations'] for t in tracks],
            rotation_step_deg=None if steps[active] is None else steps[active][0]*self.rotation_limit(stamp-old[0]),
            output_rotation_step_deg=step, continuity_segment_id=segment,
            discontinuity_reason=reset_reason, u_candidate_index=selected['u_candidate_index'])
        self.resolve_position(detection, stamp, identity_confirmed)

    @staticmethod
    def current_measurements(detection, all_branches=False):
        values = detection.get('pose_measurements', [])
        if not all_branches and not detection['pose_ambiguous'] and 'pose' in detection:
            values = [dict(pose=detection['pose'], u_candidate_index=None)]
        valid = []
        for measurement in values:
            p = measurement['pose']
            t = np.asarray(p['tvec_m'], dtype=float)
            r = np.asarray(p['rvec'], dtype=float)
            if (t.shape == (3,) and r.shape == (3,) and np.isfinite(t).all() and
                    np.isfinite(r).all() and t[2] > 0 and
                    0 <= p['rms_px'] <= 2 and 0 <= p['max_error_px'] <= 3):
                valid.append(measurement)
        return valid

    def position_evidence(self, measurements, stamp, name):
        old = self.positions.get(name)
        if old is None or old[2] < CONFIG['position_confirmation_frames'] or len(measurements) < 2:
            return None
        dt = stamp-old[0]
        if not 0 < dt <= CONFIG['pose_history_max_gap_seconds']:
            return None
        distances = [float(np.linalg.norm(np.asarray(m['pose']['tvec_m'])-old[1])) for m in measurements]
        order = np.argsort(distances)
        return dict(candidate_translation_steps_m=distances,
            translation_margin_m=distances[int(order[1])]-distances[int(order[0])],
            translation_limit_m=self.translation_limit(dt), previous_position_spread_m=old[3],
            previous_source_monotonic=old[0], selected_index=int(order[0]))

    @timed('independent_position')
    def resolve_position(self, detection, stamp, identity_confirmed):
        name = detection['pattern_id']
        detection.pop('position', None)
        evidence = dict(status='unavailable', reason='identity_unconfirmed', source_monotonic=stamp)
        detection['position_resolution'] = evidence
        if not identity_confirmed or 'pose' not in detection:
            self.positions.pop(name, None)
            if identity_confirmed:
                evidence['reason'] = 'missing_pose'
            return
        measurements = self.current_measurements(detection)
        if not measurements or detection['pose_ambiguous'] and len(measurements) < 2:
            self.positions.pop(name, None)
            evidence['reason'] = 'insufficient_current_candidates'
            return
        if detection['pose_ambiguous']:
            translations = [np.asarray(m['pose']['tvec_m']) for m in measurements]
            spread = max(float(np.linalg.norm(a-b)) for a in translations for b in translations)
            selected = min(measurements, key=lambda m: m['pose']['rms_px'])
            basis = 'current_candidate_agreement'
        else:
            spread = 0.
            selected = dict(pose=detection['pose'],
                u_candidate_index=detection['pose_resolution'].get('u_candidate_index'))
            basis = 'resolved_current_pose'
        evidence['candidate_spread_m'] = spread
        if spread > CONFIG['position_candidate_agreement_m']:
            self.positions.pop(name, None)
            evidence['reason'] = 'candidate_positions_disagree'
            return
        t = np.asarray(selected['pose']['tvec_m'], dtype=float)
        old = self.positions.get(name)
        count = 1
        if old is not None:
            dt = stamp-old[0]
            step = float(np.linalg.norm(t-old[1]))
            limit = self.translation_limit(dt)
            evidence.update(translation_step_m=step, translation_limit_m=limit)
            if 0 < dt <= CONFIG['pose_history_max_gap_seconds'] and step <= limit:
                count = old[2]+1
        self.positions[name] = (stamp, t, count, spread)
        evidence.update(confirmation_frames=count, reason='awaiting_continuous_position')
        if count >= CONFIG['position_confirmation_frames']:
            evidence.update(status='available', reason=basis)
            detection['position'] = dict(tvec_m=t.tolist(), source_monotonic=stamp,
                basis=basis, candidate_spread_m=spread, u_candidate_index=selected['u_candidate_index'])

    def resolve_pose(self, detection, stamp, identity_confirmed):
        name = detection['pattern_id']
        ambiguous = detection['pose_ambiguous']
        detection['pose_ambiguous_single_frame'] = ambiguous
        evidence = dict(status='unresolved', reason='no_trusted_anchor', anchor_ready=False,
                        source_monotonic=stamp)
        detection['pose_resolution'] = evidence
        if not identity_confirmed or 'pose' not in detection:
            self.discard(name)
            evidence['reason'] = 'identity_unconfirmed' if not identity_confirmed else 'missing_pose'
            return
        pose = detection['pose']
        anchor = self.anchors.get(name)
        if anchor is not None and not 0 < stamp-anchor[0] <= CONFIG['pose_history_max_gap_seconds']:
            self.discard(name)
            anchor = None
            evidence['reason'] = 'history_expired'
        if not ambiguous:
            evidence.update(status='single_frame', reason='unambiguous_current_image')
            pending = self.pending.get(name)
            if anchor is not None and self.bounded(anchor, pose, stamp):
                self.anchors[name] = (stamp, pose)
            elif pending is not None and self.bounded(pending, pose, stamp):
                count = pending[2]+1
                if count >= CONFIG['pose_anchor_frames']:
                    self.anchors[name] = (stamp, pose)
                    self.pending.pop(name, None)
                else:
                    self.pending[name] = (stamp, pose, count)
            else:
                self.discard_orientation(name)
                self.pending[name] = (stamp, pose, 1)
            evidence['anchor_ready'] = name in self.anchors
            return
        self.pending.pop(name, None)
        measurements = self.current_measurements(detection)
        if len(measurements) < 2:
            self.discard_orientation(name)
            evidence['reason'] = 'insufficient_current_candidates'
            return
        position = self.position_evidence(measurements, stamp, name)
        if anchor is None:
            if position is None:
                return
            evidence.update(position)
            i = position['selected_index']
            limit = position['translation_limit_m']
            other_steps = [s for j, s in enumerate(position['candidate_translation_steps_m']) if j != i]
            # Agreement history bounds its own branch uncertainty. Exclude every
            # alternative beyond the motion bound, uncertainty and a noise margin.
            if (position['candidate_translation_steps_m'][i] <= limit and
                    min(other_steps) > limit+position['previous_position_spread_m']+
                    CONFIG['position_selection_margin_m']):
                selected = measurements[i]
                excess = selected['pose']['rms_px']-min(m['pose']['rms_px'] for m in measurements)
                if excess <= CONFIG['pose_rms_excess_limit_px']:
                    self.select(detection, selected, stamp, evidence, 'position_motion_exclusion')
            return
        distances = [rotation_distance_deg(anchor[1], m['pose']) for m in measurements]
        order = np.argsort(distances)
        selected = measurements[int(order[0])]
        pose = selected['pose']
        margin = distances[int(order[1])]-distances[int(order[0])]
        position_assisted = False
        if margin < CONFIG['pose_rotation_margin_deg'] and position is not None:
            evidence.update(position)
            if position['translation_margin_m'] >= (CONFIG['position_selection_margin_m']+
                    position['previous_position_spread_m']):
                selected = measurements[position['selected_index']]
                pose = selected['pose']
                position_assisted = True
        excess = pose['rms_px']-min(m['pose']['rms_px'] for m in measurements)
        evidence.update(previous_source_monotonic=anchor[0], candidate_angles_deg=distances,
                        rotation_margin_deg=margin, selected_rms_excess_px=excess)
        if not self.bounded(anchor, pose, stamp):
            reason = 'motion_limit'
        elif margin < CONFIG['pose_rotation_margin_deg'] and not position_assisted:
            reason = 'continuity_tie'
        elif excess > CONFIG['pose_rms_excess_limit_px']:
            reason = 'current_fit_disagrees'
        else:
            self.select(detection, selected, stamp, evidence,
                'position_assisted_continuity' if position_assisted else 'temporal_continuity')
            return
        self.discard_orientation(name)
        evidence['reason'] = reason

    def select(self, detection, selected, stamp, evidence, status):
        detection.update(pose=selected['pose'], pose_ambiguous=False,
            refinement_applied=selected['refinement_applied'], robust_cost=selected['robust_cost'])
        self.anchors[detection['pattern_id']] = (stamp, selected['pose'])
        evidence.update(status=status, reason='bounded_current_candidate',
            anchor_ready=True, u_candidate_index=selected['u_candidate_index'],
            selected_candidate_stage=selected['candidate_stage'])
