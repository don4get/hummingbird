#!/usr/bin/env python3
"""
#########
Constants
#########

Constants used in other modules.

It collects enums, physical constants, etc.
"""


class PhysicalConstants:
    M = 0.0289644
    """ Molar mass of dry air [kg/mol]"""
    R = 8.31432
    """ Ideal gas constant [J/K/mol]"""
    P0 = 101325.0
    """ ISA pressure at sea level, for T=15 degrees Celsius [Pa]"""
    T0 = 288.15
    """ ISA temperature [K]"""
    L0 = -0.0065
    """ ISA Adiabatic constant [J/K/mol]"""
    g = 9.81
    """ Standard acceleration due to gravity [m/s²]"""
    absoluteZero = 273.15
    """ Absolute zero temperature [K]"""
    rho0 = 1.225
    """ Air density at sea level in ISA model"""


class StateEnum:
    """ State vector enum

    The state vector is a [12x1] vector composed of:
    NED position,
    Body velocities,
    Attitude with Euler representation,
    Attitude derivatives.

    .. todo:: Check attitude derivative precise definition
    """
    pn = 0
    """ North position [m]"""
    pe = 1
    """ East position [m]"""
    pd = 2
    """ Down position [m]"""
    u = 3
    """ X velocity in Body frame [m/s]"""
    v = 4
    """ Y velocity in Body frame [m/s]"""
    w = 5
    """ Z velocity in Body frame [m/s]"""
    phi = 6
    """ Roll angle [rad]"""
    theta = 7
    """ Pitch angle [rad]"""
    psi = 8
    """ Yaw angle [rad]"""
    p = 9
    """ Roll derivative [rad/sec]"""
    q = 10
    """ Pitch derivative [rad/sec]"""
    r = 11
    """ Yaw derivative [rad/sec]"""
    size = 12
    """ Size of the enum"""


class ActuatorEnum:
    """ Actuator vector enum

    On a fixed wing, there is 4 main actuators:
    The elevator,
    The aileron,
    The rudder,
    The throttle, named here thrust.

    The rudder is optional, not present on a flying wing.

    ..todo:: Rename thrust to throttle
    """
    elevator = 0
    """ Elevator deflection [rad]"""
    aileron = 1
    """ Aileron deflection [rad]"""
    rudder = 2
    """ Rudder deflection [rad]"""
    thrust = 3
    """ Throttle percentage [0..1]"""
    size = 4
    """ Size of the enum"""


class MotorEnum:
    """ Motor enum used for quadcopter
    """
    fl = 0
    fr = 1
    br = 2
    bl = 3


class BodyFrameEnum:
    """ Body Frame enum

    It describes the three axes of the body brame: x_body, y_body, z_body.
    """
    x = 0
    y = 1
    z = 2
