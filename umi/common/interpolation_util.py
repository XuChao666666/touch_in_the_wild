import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1]))
    return gripper_interp


class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:,:3]
        rot = st.Rotation.from_rotvec(x[:,3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose

def get_gripper_calibration_interpolator(
        aruco_measured_width, 
        aruco_actual_width):
    """
    Assumes the minimum width in aruco_actual_width
    is measured when the gripper is fully closed
    and maximum width is when the gripper is fully opened
    """
    aruco_measured_width = np.array(aruco_measured_width)
    aruco_actual_width = np.array(aruco_actual_width)
    assert len(aruco_measured_width) == len(aruco_actual_width)
    assert len(aruco_actual_width) >= 2

    aruco_min_width = np.min(aruco_actual_width)
    desired_min_gap = -0.01
    gripper_actual_width = aruco_actual_width - aruco_min_width + desired_min_gap

    # Build an interpolator mapping raw measured width -> adjusted actual width
    base_interp = get_interp1d(aruco_measured_width, gripper_actual_width)

    # Optional: print out the calibration points
    print("\n[Calibration] Creating calibration interpolator with these reference points:")
    for (meas, act) in zip(aruco_measured_width, aruco_actual_width):
        print(f"  Raw measured={meas:.4f}, Actual gap={act:.4f}")
    print(f"  => min actual width = {aruco_min_width:.4f}, so zero-shifted widths = {gripper_actual_width}\n")

    # Wrap the interpolator in a function that prints each call
    def debug_calibration_interp(raw_width):
        """
        A debug wrapper that prints raw -> calibrated
        """
        # Call the underlying interpolator
        calibrated = base_interp(raw_width)
        # Print the before/after
        print(f"[Calibration Debug] raw={raw_width:.4f} => calibrated={calibrated:.4f}")
        return calibrated

    return debug_calibration_interp
