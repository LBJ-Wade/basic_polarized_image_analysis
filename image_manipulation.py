import ehtim as eh
import numpy as np
from pmodes_simple import pmodes


#load the image, in this case an h5 file11
im = eh.image.load_image('image_ma+0.5_1990_163_10.h5')
#display it
im.display()
#blur the image with a 20uas kernel
im_blur = im.blur_circ(20*eh.RADPERUAS,20*eh.RADPERUAS)
#regrid the blurred image with a 1uas pixel size
im_blur = im_blur.regrid_image(im_blur.fovx(), int(im_blur.fovx()/eh.RADPERUAS))
#display the blurred image with polarimetric information
im_blur.display(plotp=True, plot_stokes=True)
#save the blurred image to a fits file
im_blur.save_fits('test.fits')

#now, let's compute the four image metrics from EHTC Paper VII-VIII.
#these are:
#image-integrated net linear polarization, mnet,
#image-integrated net circular polarization, vnet,
#image-averaged linear polarization fraction, mavg,
#the azimuthal Fourier mode, beta2.

#we'll compute all of these for the blurred image
mnet = im_blur.lin_polfrac()
vnet = im_blur.circ_polfrac()
#mavg doesn't have an existing builtin function, so we'll showcase the Stokes parameter vector attrs
parr = np.abs(im_blur.qvec+1j*im_blur.uvec)
mavg = np.mean(parr/im_blur.ivec)
#lastly, we'll use the imported function from the PWP modal decomposition script to compute beta2
beta2 = pmodes(im_blur, [2])[0]
#you can also compute the image net EVPA
EVPA = im_blur.evpa()


