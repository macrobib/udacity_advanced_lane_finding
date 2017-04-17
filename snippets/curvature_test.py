import numpy as np
import matplotlib.pyplot as plt

ploty = np.linspace(0, 719, num=720)
quad_coeff = 3e-4

# Random generated points +/- 50 pix of line base position.
leftx = np.array([200 + (y**2)*quad_coeff + np.random.randint(-50, high=51) for y in ploty])
rightx = np.array([900 + (y**2)*quad_coeff + np.random.randint(-50, high=51) for y in ploty])

leftx  = leftx[::-1]
rightx = rightx[::-1]

# Fit the second order polynomial.
left_fit = np.polyfit(ploty, leftx, 2)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fit = np.polyfit(ploty, rightx, 2)
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

mark_size = 3
plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis()
plt.show()

# Calculate the radius of curvature.
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)

# Convert the radius of curvature from pixel units to meters.
img_width = 700
img_height = 720
y_m_per_pix = 30/img_height
x_m_per_pix = 3.7/img_width

left_fit_cr = np.polyfit(ploty*y_m_per_pix, leftx*x_m_per_pix, 2)
right_fit_cr = np.polyfit(ploty*y_m_per_pix, rightx*x_m_per_pix, 2)

left_curverad  = ((1 + (left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])

print("Left curvature:{0} - Right curvature:{1}".format(left_curverad, right_curverad))
