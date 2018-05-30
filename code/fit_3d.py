"""
Copyright 2016 Max Planck Society, Federica Bogo, Angjoo Kanazawa. All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPLify license here:
     http://smplify.is.tue.mpg.de/license

About this Script:
============
This is a demo version of the algorithm implemented in the paper,
which fits the SMPL body model to the image given the joint detections.
The code is organized to be run on the LSP dataset.
See README to see how to download images and the detected joints.
"""

from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
from time import time
from glob import glob
import argparse

import cv2
import numpy as np
import chumpy as ch

from opendr.camera import ProjectPoints
from lib.robustifiers import GMOf
from webuser.smpl_handpca_wrapper_HAND_only import load_model
from webuser.lbs import global_rigid_transformation
from webuser.verts import verts_decorated
from render_model import render_model

_LOGGER = logging.getLogger(__name__)

# Mapping from LSP joints to SMPL joints.
# 0 Right ankle  8
# 1 Right knee   5
# 2 Right hip    2
# 3 Left hip     1
# 4 Left knee    4
# 5 Left ankle   7
# 6 Right wrist  21
# 7 Right elbow  19
# 8 Right shoulder 17
# 9 Left shoulder  16
# 10 Left elbow    18
# 11 Left wrist    20
# 12 Neck           -
# 13 Head top       added


# --------------------Camera estimation --------------------
def guess_init(model, focal_length, j2d):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model: SMPL model
    :param focal_length: camera focal length (kept fixed)
    :param j2d: 14x2 array of CNN joints
    :param init_pose: 72D vector of pose parameters used for initialization (kept fixed)
    :returns: 3D vector corresponding to the estimated camera translation
    """
    cids = [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
    # map from LSP to SMPL joints
    j2d_here = j2d[cids]
    smpl_ids = [0,13,14,15,1,2,3,4,5,6,10,11,12,7,8,9]

    opt_pose = ch.zeros(48)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])
    Jtr = Jtr[smpl_ids].r

    # 9 is L shoulder, 3 is L hip
    # 8 is R shoulder, 2 is R hip
    diff3d = np.array([Jtr[4] - Jtr[0], Jtr[13] - Jtr[0]])
    mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

    diff2d = np.array([j2d_here[4] - j2d_here[0], j2d_here[13] - j2d_here[0]])
    mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))

    est_d = focal_length * (mean_height3d / mean_height2d)
    # just set the z value
    init_t = np.array([0., 0., est_d])
    return init_t


def initialize_camera(model,
                      j2d,
                      img,
                      flength=5000.,
                      viz=False):
    """Initialize camera translation and body orientation
    :param model: SMPL model
    :param j2d: 14x2 array of CNN joints
    :param img: h x w x 3 image 
    :param init_pose: 72D vector of pose parameters used for initialization
    :param flength: camera focal length (kept fixed)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing the estimated camera,
              a boolean deciding if both the optimized body orientation and its flip should be considered,
              3D vector for the body orientation
    """
    torso_cids = [0, 5, 17]
    # # corresponding SMPL torso ids
    torso_smpl_ids = [0, 1, 7]

    center = np.array([img.shape[1] / 2, img.shape[0] / 2])

    # initialize camera rotation
    rt = ch.zeros(3)
    # initialize camera translation
    _LOGGER.info('initializing translation via similar triangles')
    init_t = guess_init(model, flength, j2d)
    t = ch.array(init_t)
    # # check how close the shoulder joints are
    try_both_orient = True

    opt_pose = ch.zeros(48)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])

    # # initialize the camera
    cam = ProjectPoints(
        f=np.array([flength, flength]), rt=rt, t=t, k=np.zeros(5), c=center)

    # # we are going to project the SMPL joints
    cam.v = Jtr

    if viz:
        viz_img = img.copy()

        # draw the target (CNN) joints
        for coord in np.around(j2d).astype(int):
            if (coord[0] < img.shape[1] and coord[0] >= 0 and
                    coord[1] < img.shape[0] and coord[1] >= 0):
                cv2.circle(viz_img, tuple(coord), 10, [0, 255, 0])

        import matplotlib.pyplot as plt
        plt.ion()

        # draw optimized joints at each iteration
        def on_step(_):
            """Draw a visualization."""
            plt.figure(1, figsize=(5, 5))
            plt.subplot(1, 1, 1)
            viz_img = img.copy()
            for coord in np.around(cam.r[torso_smpl_ids]).astype(int):
                if (coord[0] < viz_img.shape[1] and coord[0] >= 0 and
                        coord[1] < viz_img.shape[0] and coord[1] >= 0):
                    cv2.circle(viz_img, tuple(coord), 10, [0, 0, 255])
            plt.imshow(viz_img[:, :, ::-1])
            plt.draw()
            plt.show()
            plt.pause(1e-3)
    else:
        on_step = None
    # optimize for camera translation and body orientation
    free_variables = [cam.t, opt_pose[:3]]
    ch.minimize(
        # data term defined over torso joints...
        {'cam': j2d[torso_cids] - cam[torso_smpl_ids],
         # ...plus a regularizer for the camera translation
         'cam_t': 1e2 * (cam.t[2] - init_t[2])},
        x0=free_variables,
        method='dogleg',
        callback=on_step,
        options={'maxiter': 100,
                 'e_3': .0001,
                 # disp set to 1 enables verbose output from the optimizer
                 'disp': 0})
    if viz:
        plt.ioff()
    return (cam, try_both_orient, opt_pose[:3].r)


# --------------------Core optimization --------------------
def optimize_on_joints(j2d,
                       model,
                       cam,
                       img,
                       try_both_orient,
                       body_orient,
                       n_betas=10,
                       viz=False):
    """Fit the model to the given set of joints, given the estimated camera
    :param j2d: 14x2 array of CNN joints
    :param model: SMPL model
    :param cam: estimated camera
    :param img: h x w x 3 image 
    :param prior: mixture of gaussians pose prior
    :param try_both_orient: boolean, if True both body_orient and its flip are considered for the fit
    :param body_orient: 3D vector, initialization for the body orientation
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param conf: 14D vector storing the confidence values from the CNN
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing the optimized model, its joints projected on image space, the camera translation
    """
    t0 = time()
    # define the mapping LSP joints -> SMPL joints
    # cids are joints ids for LSP:
    cids = [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
    # joint ids for SMPL
    # SMPL does not have a joint for head, instead we use a vertex for the head
    # and append it later.
    smpl_ids = [0,13,14,15,1,2,3,4,5,6,10,11,12,7,8,9]

    if try_both_orient:
        flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
            cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
        flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()
        orientations = [body_orient, flipped_orient]
    else:
        orientations = [body_orient]

    if try_both_orient:
        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        errors = []

    svs = []
    cams = []
    for o_id, orient in enumerate(orientations):
        # initialize the shape to the mean shape in the SMPL training set
        betas = ch.zeros(n_betas)

        # initialize the pose by using the optimized body orientation and the
        # pose prior
        init_pose = np.hstack((orient, np.zeros(45)))

        # instantiate the model:
        # verts_decorated allows us to define how many
        # shape coefficients (directions) we want to consider (here, n_betas)
        sv = verts_decorated(
            trans=ch.zeros(3),
            pose=ch.array(init_pose),
            v_template=model.v_template,
            J=model.J_regressor,
            betas=betas,
            shapedirs=model.shapedirs[:, :, :n_betas],
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style=model.bs_style,
            f=model.f,
            bs_type=model.bs_type,
            posedirs=model.posedirs)

        # make the SMPL joints depend on betas
        Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                           for i in range(len(betas))])
        J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(
            model.v_template.r)

        # get joint positions as a function of model pose, betas and trans
        (_, A_global) = global_rigid_transformation(
            sv.pose, J_onbetas, model.kintree_table, xp=ch)
        Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

        # project SMPL joints on the image plane using the estimated camera
        cam.v = Jtr

        # data term: distance between observed and estimated joints in 2D
        obj_j2d = lambda w, sigma: (
            w * GMOf((j2d[cids] - cam[smpl_ids]), sigma))

        # joint angles pose prior, defined over a subset of pose parameters:
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        alpha = 10
        my_exp = lambda x: alpha * ch.exp(x)
        obj_angle = lambda w: w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
                                                 58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])

        if viz:
            import matplotlib.pyplot as plt
            plt.ion()

            def on_step(_):
                """Create visualization."""
                plt.figure(1, figsize=(10, 10))
                plt.subplot(1, 2, 1)
                # show optimized joints in 2D
                tmp_img = img.copy()
                for coord, target_coord in zip(
                        np.around(cam.r[smpl_ids]).astype(int),
                        np.around(j2d[cids]).astype(int)):
                    if (coord[0] < tmp_img.shape[1] and coord[0] >= 0 and
                            coord[1] < tmp_img.shape[0] and coord[1] >= 0):
                        cv2.circle(tmp_img, tuple(coord), 10, [0, 0, 255])
                    if (target_coord[0] < tmp_img.shape[1] and
                            target_coord[0] >= 0 and
                            target_coord[1] < tmp_img.shape[0] and
                            target_coord[1] >= 0):
                        cv2.circle(tmp_img, tuple(target_coord), 10,
                                   [0, 255, 0])
                plt.imshow(tmp_img[:, :, ::-1])
                plt.draw()
                plt.show()
                plt.pause(1e-2)

            on_step(_)
        else:
            on_step = None

        # weight configuration used in the paper, with joints + confidence values from the CNN
        # (all the weights used in the code were obtained via grid search, see the paper for more details)
        # the first list contains the weights for the pose priors,
        # the second list contains the weights for the shape prior
        opt_weights = zip([4.04 * 1e2],
                          [5e2])

        # run the optimization in 4 stages, progressively decreasing the
        # weights for the priors
        for stage, (w, wbetas) in enumerate(opt_weights):
            _LOGGER.info('stage %01d', stage)
            objs = {}

            objs['j2d'] = obj_j2d(1., 100)

            objs['betas'] = wbetas * betas

            ch.minimize(
                objs,
                x0=[sv.betas, sv.pose],
                method='dogleg',
                callback=on_step,
                options={'maxiter': 100,
                         'e_3': .0001,
                         'disp': 0})

        t1 = time()
        _LOGGER.info('elapsed %.05f', (t1 - t0))
        if try_both_orient:
            errors.append((objs['j2d'].r**2).sum())
        svs.append(sv)
        cams.append(cam)

    if try_both_orient and errors[0] > errors[1]:
        choose_id = 1
    else:
        choose_id = 0
    if viz:
        plt.ioff()
    print svs[choose_id].pose
    print svs[choose_id].betas
    return (svs[choose_id], cams[choose_id].r, cams[choose_id].t.r)


def run_single_fit(img,
                   j2d,
                   model,
                   n_betas=10,
                   flength=5000.,
                   scale_factor=1,
                   viz=False,
                   do_degrees=None):
    """Run the fit for one specific image.
    :param img: h x w x 3 image 
    :param j2d: 14x2 array of CNN joints
    :param conf: 14D vector storing the confidence values from the CNN
    :param model: SMPL model
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (kept fixed during optimization)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param scale_factor: int, rescale the image (for LSP, slightly greater images -- 2x -- help obtain better fits)
    :param viz: boolean, if True enables visualization during optimization
    :param do_degrees: list of degrees in azimuth to render the final fit when saving results
    :returns: a tuple containing camera/model parameters and images with rendered fits
    """
    if do_degrees is None:
        do_degrees = []

    if scale_factor != 1:
        img = cv2.resize(img, (img.shape[1] * scale_factor,
                               img.shape[0] * scale_factor))
        j2d[:, 0] *= scale_factor
        j2d[:, 1] *= scale_factor

    # estimate the camera parameters
    (cam, try_both_orient, body_orient) = initialize_camera(
        model,
        j2d,
        img,
        flength=flength,
        viz=viz)

    # fit
    (sv, opt_j2d, t) = optimize_on_joints(
        j2d,
        model,
        cam,
        img,
        try_both_orient,
        body_orient,
        n_betas=n_betas,
        viz=viz, )

    h = img.shape[0]
    w = img.shape[1]
    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])

    images = []
    orig_v = sv.r
    for deg in do_degrees:
        if deg != 0:
            aroundy = cv2.Rodrigues(np.array([0, np.radians(deg), 0]))[0]
            center = orig_v.mean(axis=0)
            new_v = np.dot((orig_v - center), aroundy)
            verts = new_v + center
        else:
            verts = orig_v
        # now render
        im = (render_model(
            verts, model.f, w, h, cam, far=20 + dist) * 255.).astype('uint8')
        images.append(im)

    # return fit parameters
    params = {'cam_t': cam.t.r,
              'f': cam.f.r,
              'pose': sv.pose.r,
              'betas': sv.betas.r}

    return params, images


def main(base_dir,
         out_dir,
         n_betas=10,
         flength=5000.,
         which_hand=False,
         viz=True):
    """Set up paths to image and joint data, saves results.
    :param base_dir: folder containing LSP images and data
    :param out_dir: output folder
    :param use_interpenetration: boolean, if True enables the interpenetration term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (an estimate)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param use_neutral: boolean, if True enables uses the neutral gender SMPL model
    :param viz: boolean, if True enables visualization during optimization
    """

    img_dir = join(abspath(base_dir), 'images')
    data_dir = join(abspath(base_dir), 'results')

    if not exists(out_dir):
        makedirs(out_dir)

    # Render degrees: List of degrees in azimuth to render the final fit.
    # Note that rendering many views can take a while.
    do_degrees = [0.]

    if not which_hand:
        model = load_model(MODEL_RIGHTHAND_PATH, ncomps=6, flat_hand_mean=False)
    else:
        model = load_model(MODEL_LEFTHAND_PATH, ncomps=6, flat_hand_mean=False)
    # Load joints
    est = np.zeros([2,21,1])
    with open(join(data_dir, 'est_joints.txt')) as f:
        lineIdx = 0
        for line in f:
            strArray = line.split(" ")
            for idx in range(0,21):
                est[:2, idx, lineIdx] = np.array([strArray[idx*3+1],strArray[idx*3+2]])
            lineIdx = lineIdx+1       



    # Load images
    img_paths = sorted(glob(join(img_dir, '*[0-9].jpg')))
    for ind, img_path in enumerate(img_paths):
        out_path = '%s/%04d.pkl' % (out_dir, ind)
        if not exists(out_path):
            _LOGGER.info('Fitting 3D body on `%s` (saving to `%s`).', img_path,
                         out_path)
            img = cv2.imread(img_path)
            if img.ndim == 2:
                _LOGGER.warn("The image is grayscale!")
                img = np.dstack((img, img, img))

            joints = est[:2, :, ind].T

            params, vis = run_single_fit(
                img,
                joints,
                model,
                n_betas=n_betas,
                flength=flength,
                scale_factor=1,
                viz=viz,
                do_degrees=do_degrees)
            if viz:
                import matplotlib.pyplot as plt
                plt.ion()
                plt.show()
                plt.subplot(121)
                plt.imshow(img[:, :, ::-1])
                if do_degrees is not None:
                    for di, deg in enumerate(do_degrees):
                        plt.subplot(122)
                        plt.cla()
                        plt.imshow(vis[di])
                        plt.draw()
                        plt.title('%d deg' % deg)
                        plt.pause(1)
                raw_input('Press any key to continue...')

            with open(out_path, 'w') as outf:
                pickle.dump(params, outf)

            # This only saves the first rendering.
            if do_degrees is not None:
                cv2.imwrite(out_path.replace('.pkl', '.png'), vis[0])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='run SMPLify on LSP dataset')
    parser.add_argument(
        'base_dir',
        default='/scratch1/projects/smplify_public/',
        nargs='?',
        help="Directory that contains images/lsp and results/lps , i.e."
        "the directory you untared smplify_code.tar.gz")
    parser.add_argument(
        '--out_dir',
        default='/tmp/smplify_lsp/',
        type=str,
        help='Where results will be saved, default is /tmp/smplify_lsp')
    parser.add_argument(
        '--no_interpenetration',
        default=False,
        action='store_true',
        help="Using this flag removes the interpenetration term, which speeds"
        "up optimization at the expense of possible interpenetration.")
    parser.add_argument(
        '--which_hand',
        default=False,
        action='store_true',
        help="Using this flag always uses the neutral SMPL model, otherwise "
        "gender specified SMPL models are used.")
    parser.add_argument(
        '--n_betas',
        default=10,
        type=int,
        help="Specify the number of shape coefficients to use.")
    parser.add_argument(
        '--flength',
        default=5000,
        type=float,
        help="Specify value of focal length.")
    parser.add_argument(
        '--side_view_thsh',
        default=25,
        type=float,
        help="This is thresholding value that determines whether the human is captured in a side view. If the pixel distance between the shoulders is less than this value, two initializations of SMPL fits are tried.")
    parser.add_argument(
        '--viz',
        default=False,
        action='store_true',
        help="Turns on visualization of intermediate optimization steps "
        "and final results.")
    args = parser.parse_args()

    # Set up paths & load models.
    # Assumes 'models' in the 'code/' directory where this file is in.
    MODEL_DIR = join(abspath(dirname(__file__)), 'models')
    # Model paths:
    MODEL_LEFTHAND_PATH = join(
        MODEL_DIR, 'MANO_LEFT.pkl')
    MODEL_RIGHTHAND_PATH = join(MODEL_DIR,
                           'MANO_RIGHT.pkl')

    main(args.base_dir, args.out_dir, args.n_betas,
         args.flength, args.which_hand, args.viz)
