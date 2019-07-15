from hummingbird.parameters.simulation_parameters import SimulationParameters


class ControlParameters:
    ts_control = SimulationParameters().dt_controller

    gravity = 9.8
    sigma = 0.05
    Va0 = 25

    # data = []
    # with open("trim.pkl", 'rb') as f:
    # data = pkl.load(f)

    # deltas_trim = data[1]

    # ----------roll loop-------------
    roll_kp = 0.4743
    roll_kd = 0.1584

    # ----------course loop-------------
    course_kp = 1.25
    course_ki = 0.2

    # ----------sideslip loop-------------
    sideslip_ki = 0
    sideslip_kp = 0.1

    # ----------yaw damper-------------
    yaw_damper_tau_r = 0.05
    yaw_damper_kp = 0.5

    # ----------pitch loop-------------
    pitch_kp = -4.5
    pitch_kd = -0.7
    K_theta_DC = 1.0

    # ----------altitude loop-------------
    altitude_kp = 0.05
    altitude_ki = 0.011
    altitude_zone = 2.0

    # ---------airspeed hold using throttle---------------
    airspeed_throttle_kp = 1.25
    airspeed_throttle_ki = 0.35

    # autopilot
    delta_e_max_deg = 45
    delta_a_max_deg = 45
    delta_r_max_deg = 45
    delta_t_max = 1.0
    delta_t_min = 0.
    roll_max_deg = 30
    pitch_max_deg = 30
    error_roll_max_deg = 15
    error_pitch_max_deg = 10
    course_omega = 0.1
    course_ksi = 3.0
    roll_ki = 0
    # roll_tau = 0.05
    roll_zeta = 1.5
    pitch_ki = 0
    # pitch_tau = 0.05
    pitch_zeta = 0.7
    airspeed_pitch_zeta = .75
    airspeed_throttle_zeta = 0.05
    altitude_omega = 0.1
    altitude_pitch_zeta = 1.0
