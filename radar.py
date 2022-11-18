import numpy as np
import cv2
import os

def load_radar(path : str):
  radar_resolution = np.array([0.0432], np.float32)
  encode_size = 5600

  raw_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  timestamps = raw_data[:, :8].copy().view(np.int64)
  azimuths = (raw_data[:, 8:10].copy().view(np.uint16) /float(encode_size) * 2 * np.pi).astype(np.float32)
  valid = raw_data[:, 10:11] == 255
  fft_data = raw_data[:, 11:].astype(np.float32)[:, :, np.newaxis]/255

  return timestamps, azimuths, valid, fft_data, radar_resolution



def radar_polar_to_cartesian(azimuths: np.ndarray, fft_data: np.ndarray, radar_resolution: float,
                            cart_resolution: float, cart_pixel_width: int, interpolate_crossover=True):

  if (cart_pixel_width % 2) == 0:
      cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
  else:
      cart_min_range = cart_pixel_width // 2 * cart_resolution
  coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
  Y, X = np.meshgrid(coords, -coords)
  sample_range = np.sqrt(Y * Y + X * X)
  sample_angle = np.arctan2(Y, X)
  sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

  # Interpolate Radar Data Coordinates
  azimuth_step = azimuths[1] - azimuths[0]
  sample_u = (sample_range - radar_resolution / 2) / radar_resolution
  sample_v = (sample_angle - azimuths[0]) / azimuth_step

  # We clip the sample points to the minimum sensor reading range so that we
  # do not have undefined results in the centre of the image. In practice
  # this region is simply undefined.
  sample_u[sample_u < 0] = 0

  if interpolate_crossover:
      fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
      sample_v = sample_v + 1

  polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
  cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
  return cart_img


def play(dir):
  timestamps_path = os.path.join(os.path.join(dir, 'radar.timestamps'))
  print(timestamps_path)
  if not os.path.isfile(timestamps_path):
      raise IOError("Could not find timestamps file")

  # Cartesian Visualsation Setup
  # Resolution of the cartesian form of the radar scan in metres per pixel
  cart_resolution = .25
  # Cartesian visualisation size (used for both height and width)
  cart_pixel_width = 501  # pixels
  interpolate_crossover = True

  radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
  time = 0


  for i in range(0, len(radar_timestamps)):
    filename = os.path.join(dir, "radar", str(radar_timestamps[i]) + '.png')

    if i == (len(radar_timestamps)-1):
      time = 10
    else:
      time = int((radar_timestamps[i+1] - radar_timestamps[i])/1000)

    if not os.path.isfile(filename):
      raise FileNotFoundError("Could not find the file {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                    interpolate_crossover)

    downsample_rate = 4

    fft_data_vis = fft_data[:, ::downsample_rate]
    resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
    fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
    vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))

    cv2.imshow("Image", vis*2. )  # The data is doubled to improve visualisation
    cv2.waitKey(time)

