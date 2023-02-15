import matplotlib.pyplot as plt

import Utility as util

img = util.load_img('data/Images/Chambre/IMG_6567.JPG')
ref = util.load_img('data/Images/Chambre/Reference.JPG')
sub = util.get_subtracted_threshold(img, ref, 20)

plt.figure()
plt.imshow(sub, cmap='Greys_r',interpolation='none')
plt.show()

util.quick_plot2(sub, ref, 'Subtracted', 'Reference')