## This script generates an animation comparing the intensity profiles of a Laguerre-Gaussian beam (LG01) and a fundamental Gaussian beam (TEM)
## Abrar Shafin
## Date: 25 Aug 2025

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from LightPipes import *

def beam_profiles(size,flag):
  """
  Function creates the canvas of a beam profile
  
  INPUTS:
  - size= size x size plot							            [FLOAT]
  - flag= normalised, unnormalised, 8-bit heatmaps	[0,1,2]
  
  OUTPUTS:
  - F = Field of the beams							            [LIGHTPIPES FIELD]
  - I_LG = Intensity Field of the LG beam			      [LIGHTPIPES FIELD]
  - I_TEM = Intensity Field of Gaussian			      [LIGHTPIPES FIELD]
  """
  wavelength=633*nm
  N=500
  w0 = 0.7*mm
  
  F=Begin(size,wavelength,N)
  F=[GaussBeam(F, w0 ,doughnut=True,m=1), GaussBeam(F, w0)] # LG0,1 Laguerre Gauss beam
  I_LG = Intensity(flag,F[0])
  I_TEM = Intensity(flag,F[1])
  return I_LG, I_TEM

def intensity_plots(I_LG, I_TEM):
  """
  Function to create figures & axes for a three-plot figure describing intensity profiles
  which has artists for animation
  
  INPUTS:
  - I_LG = Intensity Field of the LG beam			      [LIGHTPIPES FIELD]
  - I_TEM = Intensity Field of Gaussian			      [LIGHTPIPES FIELD]
  
  OUTPUTS:
  - fig = figure handle of plot 					          [FIGURE]
  - profile_lg = profiling handle for LG beam			  [AXES]
  - profile_tem = profiling handle for Gaussian			[AXES]
  - line_TEM = scanning line handle for Image     [AXES]
  - line_LG = scanning line handle for Image  			[AXES]
  """
  fig = plt.figure(figsize=(12, 10))
  fig.set_tight_layout(True)
  gs = gridspec.GridSpec(2, 2)
  ax1 = plt.subplot(gs[0, 0])
  ax2 = plt.subplot(gs[0, 1])
  ax3 = plt.subplot(gs[1, :])

  im1 = ax1.imshow(I_LG, cmap='binary', aspect='equal')
  line_LG, = ax1.plot([0, I_LG.shape[1]], [0, 0], 'c', lw=2)
  ax1.axis('off')

  im2 = ax2.imshow(I_TEM, cmap='binary')
  line_TEM, = ax2.plot([0, I_TEM.shape[1]], [0, 0], 'g', lw=2)
  ax2.axis('off')

  profile_lg, = ax3.plot([], [], c='c', lw=2, label='$LG_{01}$')
  profile_tem, = ax3.plot([], [], c='g', lw=2, label='$TEM_{00}$')
  ax3.set_xlim(0, I_LG.shape[1])
  ax3.set_ylim(0, max(I_LG.max(), I_TEM.max())*1.1)
  ax3.grid()
  ax3.set_xlabel('Picture Width, [pixel]', fontsize=16)
  ax3.set_ylabel('Intensity, [8-bit]', fontsize=16)
  ax3.legend(fontsize=16)
  ax3.set_title('Intensity Profiles of $LG_{01}$ and $TEM_{00}$ modes')

  return fig, profile_lg, profile_tem, line_TEM, line_LG

def main():
  #%% [CALL THE INTENSITY PROFILES]
  I_LG, I_TEM = beam_profiles(3*mm,2)
  
  #%% [SETUP INITIAL PLOT & AXES]
  fig, profile_lg, profile_tem, line_TEM, line_LG = intensity_plots(I_LG, I_TEM)
  
  #%% [MIDPOINT SLICE]
  mid = I_LG.shape[0] // 2
  x = np.arange(I_LG.shape[1])

  # Update the line positions on both heatmaps
  line_LG.set_ydata([mid, mid])
  line_TEM.set_ydata([mid, mid])

  # Set the profile data for the midpoint
  profile_lg.set_data(x, I_LG[mid, :])
  profile_tem.set_data(x, I_TEM[mid, :])

  # Compute gradients (derivatives)
  grad_LG = np.gradient(I_LG[mid, :])
  grad_TEM = np.gradient(I_TEM[mid, :])

  #%% [SET PROFILE DATA]
  profile_lg.set_data(x, I_LG[mid, :])
  profile_tem.set_data(x, I_TEM[mid, :])

  #%% [PLOT GRADIENTS ON SAME AXES]
  ax3_grad = profile_lg.axes.twinx()  # reuse the same bottom plot from intensity_plots()
  grad_LG, = ax3_grad.plot(x, grad_LG, 'c--', lw=1.5, label=r'$\nabla LG_{01}$')
  grad_TEM, = ax3_grad.plot(x, grad_TEM, 'g--', lw=1.5, label=r'$\nabla TEM_{00}$')
  ax3_grad.set_ylabel('Magnitude of Gradients', fontsize=16)

  # Update title and legend to reflect new content
  lines = [profile_lg, profile_tem, grad_LG, grad_TEM]

  labels = [l.get_label() for l in lines]
   
  profile_lg.axes.set_title('Midpoint Intensity and Gradient Profiles of $LG_{01}$ and $TEM_{00}$')
  profile_lg.axes.legend(lines, labels, fontsize=14)

  #%% [FINAL TOUCHES AND SAVE]
  plt.tight_layout()
  plt.savefig('beam_profiles_midpoint.png', dpi=300)
  plt.show()

  print("Saved static midpoint comparison as 'beam_profiles_midpoint.png'")

  #%% [ANIMATE]
  
  def animate(i):
      line_LG.set_ydata([i,i])    											# update the horizontal line position from staring at ith row to ending at ith row
      line_TEM.set_ydata([i,i])   										# update the horizontal line position from staring at ith row to ending at ith row
      profile_lg.set_data(np.arange(I_LG.shape[1]), I_LG[i, :])   			# update the intensity profile data
      profile_tem.set_data(np.arange(I_TEM.shape[1]), I_TEM[i, :])    	# update the intensity profile data

      # give a progress bar in regards to total number of frames, i+1 because i starts from 0
      print(f'Progress: {round((i+1)/I_LG.shape[0]*100,2)}%', end='\r')
      return line_LG, line_TEM, profile_lg, profile_tem

  ani = FuncAnimation(fig, animate, frames=I_LG.shape[0], interval=30, blit=True)
  ani.save('beam_profiles.gif', writer='ffmpeg', dpi=200)
  print()
  print('Done!')
  return None

if __name__ == "__main__":
  main()
