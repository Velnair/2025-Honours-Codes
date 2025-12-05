## This script is to generate a gif overlaying the FLIR tracked path on the EB tracked objects
## The FLIR image is scaled to the EB plot, with the option to offset the image to center the cross section with the scatter points
## Abrar Shafin
## Date: 9 Sep 2025

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import sys

# import the EB coords.txt & set labels to the data
# import the tracked FLIR pathline, centering it at (0,0). Note symmetric centering would not work as the cameras may be offset
# Calculate the aspect ratio of the image

def call_info():
  """
  Obtain the required FLIR image & EB tracking data from .txt file
  """

  if len(sys.argv) < 5:
    print("Usage: python EB-FLIR_comp.py <FLIR_tracked_path> <EB.txt_boxes_path> <(x_offset, y_offset)`>")
    sys.exit(1)

  #%% [EXTRACT FLIR TRACKED PATH IMAGE]
  img_path = sys.argv[1]
  FLIR = cv2.imread(img_path)
  FLIR = cv2.flip(FLIR, 1)  # flip image horizontally, translations were mirrored due to beam spliter setup
  
  if FLIR is None:		# failure to read image
      print(f"Error: Could not read image '{img_path}'")
      sys.exit(1)

  #%% [EXTRACT EB BOUNDING BOXES DATA]
  data_path = sys.argv[2]
  EB = np.loadtxt(data_path)

  if EB is None:		# failure to read .txt
    print(f"Error: Could not read .txt '{data_path}'")
    sys.exit(1)
  
  #%% [OBTAIN OFFSET COORDINATES]
  try:
    x_off = int(sys.argv[3])
    y_off = int(sys.argv[4])
  except ValueError:
    print(f"Error: Coordinates must be integers. Received: '{sys.argv[3]}', '{sys.argv[4]}'")
    sys.exit(1)

  return FLIR, EB, x_off, y_off

def EB_figure(EB_s, FLIR_s,x_off,y_off,FLIR):
  """
  Function to correctly scale the FLIR image into the EB plot for data overlay
  
  INPUTS:
  - EB_s = pix -> m conversion, EBC       [FLOAT]
  - FLIR_s = pix -> m conversion, FLIR    [FLOAT]
  - x_off = x offset to center image		  [INT]
  - y_off = y offset to center image		  [INT]
  - FLIR = FLIR image of data				      [TUPLE]
  
  OUTPUTS:
  - fig = figure handle of plot 			    [FIGURE]
  - ax = axis handle of plot				      [AXES]
  """
  
  #%% [SET UP PLOT 1280 X 720]
  fig, ax = plt.subplots(figsize=(10, 6))
  fig.set_tight_layout(True)
  ax.set_xlim(0, 1280*EB_s)   # width, physical measurement
  ax.set_ylim(0, 720*EB_s)    # height, physical measurement
  ax.set_title('FLIR Tracked Path on EB Tracked Objects')
  ax.set_xlabel('Width, [$mm$]')
  ax.set_ylabel('Height, [$mm$]')
  ax.grid(True)

  frame_w, frame_h = (1280 - x_off)*EB_s, (720 - y_off)*EB_s
  frame_aspect = frame_w / frame_h

  #%% [FIT IMAGE WITHIN FRAME WHILST ENFORCING ASPECT RATIO TO AVOID SKEWING OF FLIR-IMAGE]
  
  img = FLIR
  img_h, img_w = img.shape[0]*FLIR_s, img.shape[1]*FLIR_s   # image dimensions, physical measurement
  aspect_ratio = img_w / img_h                          # aspect ratio after scaling
  
  if aspect_ratio > frame_aspect:
    # Image is wider than frame, fit width
    new_w = frame_w
    new_h = new_w / aspect_ratio
    x_min, x_max = 0, new_w
    y_center = frame_h / 2
    y_min = y_center - new_h / 2
    y_max = y_center + new_h / 2
  else:
    # Image is taller than frame, fit height
    new_h = frame_h
    new_w = new_h * aspect_ratio
    y_min, y_max = 0, new_h
    x_center = frame_w / 2
    x_min = x_center - new_w / 2
    x_max = x_center + new_w / 2

  ax.imshow(img, extent=[x_min, x_max, y_min, y_max])

  return fig, ax

def main():
  
  #%% [CALL IN DATA & SCALE ACCORDINGLY]
  EB_s = 4.86e-3  #millimeters
  FLIR_s = 6.9e-3  #millimeters
  
  FLIR,EB,x_off,y_off = call_info()
  
  # identify all coloumns
  x, y = EB[:, 0]*EB_s, EB[:, 1]*EB_s   # bounding box center coordinates
  t = EB[:, 2]
  w, h = EB[:, 5], EB[:, 6]   # bounding box width & height
  obj_id = EB[:, 7]  # object ID
  ev_id = EB[:, 8]  # event ID
  no_objs = EB[:, 9]  # number of objects in the event
  
  #%% [CREATE EB-IMAGE WITH FLIR IMAGE CENTERED IN]
  fig, ax = EB_figure(EB_s,FLIR_s,x_off,y_off,FLIR)
  
  #%% [CREATE A FUNCTION TO OBTAIN A VECTOR ]
  frames = 120
  alphas = np.concatenate([
      np.linspace(0, 1, frames//2, endpoint=False),
      np.linspace(1, 0, frames//2, endpoint=False)
  ])
  scatter = ax.scatter(x,y,c=obj_id,cmap='brg',alpha=alphas[0])
  
  # save test figs for debugging
  scatter.set_alpha(1)
  fig.savefig('test')
  
  print('Test figs: Done!')
  
  #%% [ANIMATE]
  def animate(i):
    scatter.set_alpha(alphas[i])
    print(f'Progress: {round((i+1)/frames*100,2)}%', end='\r')
    return None
    
  ani = FuncAnimation(fig, animate, frames=frames, interval=30, blit=False)
  ani.save('scatter.gif', writer='ffmpeg', dpi=200)
  plt.show()
  print('\nDone!')
  
  return None

# run the code when called
if __name__ == "__main__":
    main()
