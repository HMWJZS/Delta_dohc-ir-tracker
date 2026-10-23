"""Opt-in inclusive/exclusive wall-time spans; no image/pose changes."""
from contextlib import contextmanager
from functools import wraps
import os
import threading
import time

_local = threading.local()


def enabled():
    return os.environ.get('DOHC_IR_TRACKER_PROFILE', '').lower() in ('1', 'true', 'yes')


def active():
    return getattr(_local, 'trace', None) is not None


@contextmanager
def span(name):
    trace = getattr(_local, 'trace', None)
    if trace is None:
        yield
        return
    item = [name, time.perf_counter_ns(), 0]
    trace['stack'].append(item)
    try:
        yield
    finally:
        elapsed = time.perf_counter_ns()-item[1]
        trace['stack'].pop()
        if trace['stack']:
            trace['stack'][-1][2] += elapsed
        value = trace['stages'].setdefault(name, dict(calls=0, inclusive_ms=0., exclusive_ms=0.))
        value['calls'] += 1
        value['inclusive_ms'] += elapsed/1e6
        value['exclusive_ms'] += (elapsed-item[2])/1e6


def call(name, function, *args, **kwargs):
    if getattr(_local, 'trace', None) is None:
        return function(*args, **kwargs)
    with span(name):
        return function(*args, **kwargs)


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if getattr(_local, 'trace', None) is None:
                return function(*args, **kwargs)
            with span(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def note(name, value):
    trace = getattr(_local, 'trace', None)
    if trace is not None:
        trace.setdefault('details', {}).setdefault(name, []).append(value)


def profiled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not enabled():
            return function(*args, **kwargs)
        previous = getattr(_local, 'trace', None)
        trace = dict(stack=[], stages={})
        _local.trace = trace
        try:
            with span('detect_total'):
                result = function(*args, **kwargs)
            result['_profile'] = dict(schema='dohc.infrared.profile.v1',
                clock='perf_counter_ns', stages=trace['stages'], details=trace.get('details', {}))
            return result
        finally:
            _local.trace = previous
    return wrapped
