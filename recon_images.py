"""=============================================================================
Qualitative experiment showing better image reconstructions given more sampling.
============================================================================="""

from   imageio import imwrite
import matplotlib.pyplot as plt
from   matplotlib import rc
import numpy as np
from   rsvd import rsvd

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': 'Arial',
    'size' : 16
})
rc('text', usetex=False)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    fig, axarr = plt.subplots(1, 4)
    fig.set_size_inches(15, 4)
    
    from PIL import Image #MY CODE
    A = np.array(Image.open('Nott.png').convert('L')) #MY CODE

    imwrite('figures/Nott.jpg', A.astype(np.uint8))
    rank = 15

    U, S, Vt = np.linalg.svd(A)
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
    M_recon = (U * S) @ Vt
    fname = 'figures/Nott_actual_k%s.jpg' % rank
    imwrite(fname, M_recon.astype(np.uint8))
    axarr[0].imshow(M_recon, cmap='gray')
    axarr[0].axis('off')
    axarr[0].set_title(r'SVD')

    for i, l in enumerate([1, 10, 100]):
        U, S, Vt = rsvd(A, rank, l, n_subspace_iters=0)
        M_recon = (U * S) @ Vt
        fname = 'figures/Nott_k%s_p%s.jpg' % (rank, l)
        imwrite(fname, M_recon.astype(np.uint8))
        axarr[i+1].imshow(M_recon, cmap='gray')
        axarr[i+1].axis('off')
        axarr[i+1].set_title(r'RSVD, $\ell = %s$' % l)

    # plt.title('foo')
    plt.tight_layout()
    plt.show()
