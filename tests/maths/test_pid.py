from hummingbird.maths.pid import *


def test_pid_init():
    pid = Pid(kp=1., ki=2., kd=3., dt=4., sigma=5., limit=6., lower_limit=7.)

    assert pid.kp == 1.
    assert pid.ki == 2.
    assert pid.kd == 3.
    assert pid.dt == 4.
    assert pid.sigma == 5.
    assert pid.upper_limit == 6.
    assert pid.lower_limit == 7.


def test_pid_update_proportional():
    pid = Pid(kp=1., ki=0., kd=0., dt=0., sigma=0., limit=1.)
    reference = 1.
    feedback = 0.
    output = pid(reference, feedback)
    expected_output = pid.kp * (reference - feedback)
    assert output == expected_output


def test_pid_update_proportional_saturated():
    pid = Pid(kp=1., ki=0., kd=0., dt=0., sigma=0., limit=1.)
    reference = 2.
    feedback = 0.
    output = pid(reference, feedback)
    expected_output = pid.upper_limit
    assert output == expected_output


def test_pid_update_integral():
    pid = Pid(kp=0., ki=1., kd=0., dt=1., sigma=0., limit=2.)
    reference = 1.
    feedback = 0.
    output = pid(reference, feedback)
    expected_output = pid.ki * (reference - feedback) / 2.
    assert output == expected_output
    output = pid(reference, feedback)
    expected_output += pid.ki * (reference - feedback)
    assert output == expected_output


def test_pid_update_integral_saturated():
    pid = Pid(kp=0., ki=1., kd=0., dt=1., sigma=0., limit=2.)
    reference = 5.
    feedback = 0.
    output = pid(reference, feedback)
    expected_output = pid.upper_limit
    assert output == expected_output
    assert pid.integrator == expected_output


def test_pid_update_derivative_numerical_derivative():
    pid = Pid(kp=0., ki=0., kd=1., dt=1., sigma=0., limit=100.)
    reference = 5.
    feedback = 1.
    output = pid(reference, feedback)
    expected_output = 2. * feedback
    assert output == expected_output


def test_pid_update_derivative():
    pid = Pid(kp=0., ki=0., kd=1., dt=1., sigma=0., limit=100.)
    reference = 5.
    feedback = 1.
    derivative = 1.
    output = pid(reference, feedback, derivative)
    expected_output = derivative
    assert output == expected_output


def test_pid_update_derivative_saturated():
    pid = Pid(kp=0., ki=0., kd=1., dt=1., sigma=0., limit=2.)
    reference = 5.
    feedback = 1.
    derivative = 3.
    output = pid(reference, feedback, derivative)
    expected_output = pid.upper_limit
    assert output == expected_output