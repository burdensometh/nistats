"""
Test the second level model.
"""
from __future__ import with_statement

import os

import numpy as np
import pandas as pd
import warnings

import pytest

from nibabel import (load,
                     Nifti1Image,
                     save,
                     )
from nibabel.tmpdirs import InTemporaryDirectory
from nilearn.image import concat_imgs
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           )

from nistats.first_level_model import (FirstLevelModel,
                                       run_glm,
                                       )
from nistats.second_level_model import (SecondLevelModel,
                                        non_parametric_inference,
                                        _cluster_robust_contrast_variance,
                                        _cluster_design_quantities,
                                        )
from nistats.regression import OLSModel
from nistats._utils.testing import _write_fake_fmri_data
from nistats.utils import get_data

# This directory path
BASEDIR = os.path.dirname(os.path.abspath(__file__))
FUNCFILE = os.path.join(BASEDIR, 'functional.nii.gz')


def test_high_level_glm_with_paths():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask)
        # asking for contrast before model fit gives error
        with pytest.raises(ValueError):
            model.compute_contrast([])
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        model = model.fit(Y, design_matrix=X)
        c1 = np.eye(len(model.design_matrix_.columns))[0]
        z_image = model.compute_contrast(c1, output_type='z_score')
        assert isinstance(z_image, Nifti1Image)
        assert_array_equal(z_image.affine, load(mask).affine)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del Y, FUNCFILE, func_img, model


def test_high_level_non_parametric_inference_with_paths():
    with InTemporaryDirectory():
        n_perm = 100
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        c1 = np.eye(len(X.columns))[0]
        neg_log_pvals_img = non_parametric_inference(Y, design_matrix=X,
                                                     second_level_contrast=c1,
                                                     mask=mask, n_perm=n_perm)
        neg_log_pvals = get_data(neg_log_pvals_img)

        assert isinstance(neg_log_pvals_img, Nifti1Image)
        assert_array_equal(neg_log_pvals_img.affine, load(mask).affine)

        assert np.all(neg_log_pvals <= - np.log10(1.0 / (n_perm + 1)))
        assert np.all(0 <= neg_log_pvals)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory
        del X, Y, FUNCFILE, func_img, neg_log_pvals_img


def test_fmri_inputs():
    # Test processing of FMRI inputs
    with InTemporaryDirectory():
        # prepare fake data
        p, q = 80, 10
        X = np.random.randn(p, q)
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        T = func_img.shape[-1]
        des = pd.DataFrame(np.ones((T, 1)), columns=['a'])
        des_fname = 'design.csv'
        des.to_csv(des_fname)

        # prepare correct input first level models
        flm = FirstLevelModel(subject_label='01').fit(FUNCFILE,
                                                      design_matrices=des)
        flms = [flm, flm, flm]
        # prepare correct input dataframe and lists
        shapes = ((7, 8, 9, 1),)
        _, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]

        dfcols = ['subject_label', 'map_name', 'effects_map_path']
        dfrows = [['01', 'a', FUNCFILE], ['02', 'a', FUNCFILE],
                  ['03', 'a', FUNCFILE]]
        niidf = pd.DataFrame(dfrows, columns=dfcols)
        niimgs = [FUNCFILE, FUNCFILE, FUNCFILE]
        niimg_4d = concat_imgs(niimgs)
        confounds = pd.DataFrame([['01', 1], ['02', 2], ['03', 3]],
                                 columns=['subject_label', 'conf1'])
        sdes = pd.DataFrame(X[:3, :3], columns=['intercept', 'b', 'c'])

        # smoke tests with correct input
        # First level models as input
        SecondLevelModel(mask_img=mask).fit(flms)
        SecondLevelModel().fit(flms)
        # Note : the following one creates a singular design matrix
        SecondLevelModel().fit(flms, confounds)
        SecondLevelModel().fit(flms, None, sdes)
        # dataframes as input
        SecondLevelModel().fit(niidf)
        SecondLevelModel().fit(niidf, confounds)
        SecondLevelModel().fit(niidf, confounds, sdes)
        SecondLevelModel().fit(niidf, None, sdes)
        # niimgs as input
        SecondLevelModel().fit(niimgs, None, sdes)
        # 4d niimg as input
        SecondLevelModel().fit(niimg_4d, None, sdes)

        # test wrong input errors
        # test first level model requirements
        with pytest.raises(ValueError):
            SecondLevelModel().fit(flm)
        with pytest.raises(ValueError):
            SecondLevelModel().fit([flm])
        # test dataframe requirements
        with pytest.raises(ValueError):
            SecondLevelModel().fit(niidf['subject_label'])
        # test niimgs requirements
        with pytest.raises(ValueError):
            SecondLevelModel().fit(niimgs)
        with pytest.raises(ValueError):
            SecondLevelModel().fit(niimgs + [[]],
                                   confounds)
        # test first_level_conditions, confounds, and design
        with pytest.raises(ValueError):
            SecondLevelModel().fit(flms, ['', []])
        with pytest.raises(ValueError):
            SecondLevelModel().fit(flms, [])
        with pytest.raises(ValueError):
            SecondLevelModel().fit(flms, confounds['conf1'])
        with pytest.raises(ValueError):
            SecondLevelModel().fit(flms, None, [])


def test_fmri_inputs_for_non_parametric_inference():
    # Test processing of FMRI inputs
    with InTemporaryDirectory():
        # prepare fake data
        p, q = 80, 10
        X = np.random.randn(p, q)
        shapes = ((7, 8, 9, 10),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        T = func_img.shape[-1]
        des = pd.DataFrame(np.ones((T, 1)), columns=['a'])
        des_fname = 'design.csv'
        des.to_csv(des_fname)

        # prepare correct input first level models
        flm = FirstLevelModel(subject_label='01').fit(FUNCFILE,
                                                      design_matrices=des)
        # prepare correct input dataframe and lists
        shapes = ((7, 8, 9, 1),)
        _, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]

        dfcols = ['subject_label', 'map_name', 'effects_map_path']
        dfrows = [['01', 'a', FUNCFILE], ['02', 'a', FUNCFILE],
                  ['03', 'a', FUNCFILE]]
        niidf = pd.DataFrame(dfrows, columns=dfcols)
        niimgs = [FUNCFILE, FUNCFILE, FUNCFILE]
        niimg_4d = concat_imgs(niimgs)
        confounds = pd.DataFrame([['01', 1], ['02', 2], ['03', 3]],
                                 columns=['subject_label', 'conf1'])
        sdes = pd.DataFrame(X[:3, :3], columns=['intercept', 'b', 'c'])

        # test missing second-level contrast
        # niimgs as input
        with pytest.raises(ValueError):
            non_parametric_inference(niimgs, None, sdes)
        with pytest.raises(ValueError):
            non_parametric_inference(niimgs, confounds, sdes)
        # 4d niimg as input
        with pytest.raises(ValueError):
            non_parametric_inference(niimg_4d, None, sdes)

        # test wrong input errors
        # test first level model
        with pytest.raises(ValueError):
            non_parametric_inference(flm)
        # test list of less than two niimgs
        with pytest.raises(ValueError):
            non_parametric_inference([FUNCFILE])
        # test dataframe
        with pytest.raises(ValueError):
            non_parametric_inference(niidf)
        # test niimgs requirements
        with pytest.raises(ValueError):
            non_parametric_inference(niimgs)
        with pytest.raises(ValueError):
            non_parametric_inference(niimgs + [[]], confounds)
        with pytest.raises(ValueError):
            non_parametric_inference([FUNCFILE])
        # test other objects
        with pytest.raises(ValueError):
            non_parametric_inference('random string object')
        del X, FUNCFILE, func_img


def _first_level_dataframe():
    names = ['con_01', 'con_01', 'con_01']
    subjects = ['01', '02', '03']
    maps = ['', '', '']
    dataframe = pd.DataFrame({'map_name': names,
                              'subject_label': subjects,
                              'effects_map_path': maps})
    return dataframe


def test_second_level_model_glm_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask)
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])

        model = model.fit(Y, design_matrix=X)
        model.compute_contrast()
        labels1 = model.labels_
        results1 = model.results_

        labels2, results2 = run_glm(
            model.masker_.transform(Y), X.values, 'ols')
        assert_almost_equal(labels1, labels2, decimal=1)
        assert len(results1) == len(results2)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del func_img, FUNCFILE, model, X, Y


def test_non_parametric_inference_permutation_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)

        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])

        neg_log_pvals_img = non_parametric_inference(Y, design_matrix=X,
                                                     mask=mask, n_perm=100)

        assert get_data(neg_log_pvals_img).shape == shapes[0][:3]
        del func_img, FUNCFILE, neg_log_pvals_img, X, Y


def test_second_level_model_contrast_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask)
        # asking for contrast before model fit gives error
        with pytest.raises(ValueError):
            model.compute_contrast('intercept')
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        model = model.fit(Y, design_matrix=X)
        ncol = len(model.design_matrix_.columns)
        c1, cnull = np.eye(ncol)[0, :], np.zeros(ncol)
        # smoke test for different contrasts in fixed effects
        model.compute_contrast(c1)
        z_image = model.compute_contrast(c1, output_type='z_score')
        stat_image = model.compute_contrast(c1, output_type='stat')
        p_image = model.compute_contrast(c1, output_type='p_value')
        effect_image = model.compute_contrast(c1, output_type='effect_size')
        variance_image = \
            model.compute_contrast(c1, output_type='effect_variance')

        # Test output_type='all', and verify images are equivalent
        all_images = model.compute_contrast(c1, output_type='all')
        assert_array_equal(get_data(all_images['z_score']),
                           get_data(z_image))
        assert_array_equal(get_data(all_images['stat']),
                           get_data(stat_image))
        assert_array_equal(get_data(all_images['p_value']),
                           get_data(p_image))
        assert_array_equal(get_data(all_images['effect_size']),
                           get_data(effect_image))
        assert_array_equal(get_data(all_images['effect_variance']),
                           get_data(variance_image))

        # formula should work (passing variable name directly)
        model.compute_contrast('intercept')
        # or simply pass nothing
        model.compute_contrast()
        # passing null contrast should give back a value error
        with pytest.raises(ValueError):
            model.compute_contrast(cnull)
        # passing wrong parameters
        with pytest.raises(ValueError):
            model.compute_contrast([])
        with pytest.raises(ValueError):
            model.compute_contrast(c1, None, '')
        with pytest.raises(ValueError):
            model.compute_contrast(c1, None, [])
        with pytest.raises(ValueError):
            model.compute_contrast(c1, None, None, '')
        # check that passing no explicit contrast when the design
        # matrix has more than one columns raises an error
        X = pd.DataFrame(np.random.rand(4, 2), columns=['r1', 'r2'])
        model = model.fit(Y, design_matrix=X)
        with pytest.raises(ValueError):
            model.compute_contrast(None)
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del func_img, FUNCFILE, model, X, Y


def test_non_parametric_inference_contrast_computation():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # asking for contrast before model fit gives error
        with pytest.raises(ValueError):
            non_parametric_inference(None, None, None, 'intercept', mask)
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        # formula should work without second-level contrast
        neg_log_pvals_img = non_parametric_inference(Y, design_matrix=X,
                                                     mask=mask, n_perm=100)

        ncol = len(X.columns)
        c1, cnull = np.eye(ncol)[0, :], np.zeros(ncol)
        # formula should work with second-level contrast
        neg_log_pvals_img = non_parametric_inference(Y, design_matrix=X,
                                                     second_level_contrast=c1,
                                                     mask=mask, n_perm=100)
        # formula should work passing variable name directly
        neg_log_pvals_img = \
            non_parametric_inference(Y, design_matrix=X,
                                     second_level_contrast='intercept',
                                     mask=mask, n_perm=100)

        # passing null contrast should give back a value error
        with pytest.raises(ValueError):
            non_parametric_inference(Y, X, cnull, 'intercept', mask)
        # passing wrong parameters
        with pytest.raises(ValueError):
            non_parametric_inference(Y, X, [], 'intercept', mask)
        # check that passing no explicit contrast when the design
        # matrix has more than one columns raises an error
        X = pd.DataFrame(np.random.rand(4, 2), columns=['r1', 'r2'])
        with pytest.raises(ValueError):
            non_parametric_inference(Y, X, None)
        del func_img, FUNCFILE, neg_log_pvals_img, X, Y


def test_second_level_model_contrast_computation_with_memory_caching():
    with InTemporaryDirectory():
        shapes = ((7, 8, 9, 1),)
        mask, FUNCFILE, _ = _write_fake_fmri_data(shapes)
        FUNCFILE = FUNCFILE[0]
        func_img = load(FUNCFILE)
        # Ordinary Least Squares case
        model = SecondLevelModel(mask_img=mask, memory='nilearn_cache')
        # fit model
        Y = [func_img] * 4
        X = pd.DataFrame([[1]] * 4, columns=['intercept'])
        model = model.fit(Y, design_matrix=X)
        ncol = len(model.design_matrix_.columns)
        c1 = np.eye(ncol)[0, :]
        # test memory caching for compute_contrast
        model.compute_contrast(c1, output_type='z_score')
        # or simply pass nothing
        model.compute_contrast()
        # Delete objects attached to files to avoid WindowsError when deleting
        # temporary directory (in Windows)
        del func_img, FUNCFILE, model, X, Y


def test_param_mask_deprecation_SecondLevelModel():
    """ Tests whether use of deprecated keyword parameter `mask`
    raises the correct warning & transfers its value to
    replacement parameter `mask_img` correctly.
    """
    deprecation_msg = (
        'The parameter "mask" will be removed in next release of Nistats. '
        'Please use the parameter "mask_img" instead.'
    )
    mask_filepath = '~/masks/mask_01.nii.gz'
    with warnings.catch_warnings(record=True) as raised_warnings:
        slm1 = SecondLevelModel(mask=mask_filepath)
        slm2 = SecondLevelModel(mask_img=mask_filepath)
        slm3 = SecondLevelModel(mask_filepath)
    assert slm1.mask_img == mask_filepath
    assert slm2.mask_img == mask_filepath
    assert slm3.mask_img == mask_filepath

    with pytest.raises(AttributeError):
        slm1.mask == mask_filepath
    with pytest.raises(AttributeError):
        slm2.mask == mask_filepath
    with pytest.raises(AttributeError):
        slm3.mask == mask_filepath

    raised_param_deprecation_warnings = [
        raised_warning_ for raised_warning_
        in raised_warnings if
        str(raised_warning_.message).startswith('The parameter')
        ]

    assert len(raised_param_deprecation_warnings) == 1
    for param_warning_ in raised_param_deprecation_warnings:
        assert str(param_warning_.message) == deprecation_msg
        assert param_warning_.category is DeprecationWarning


def _cluster_rng():
    return np.random.RandomState(42)


def _make_effect_imgs(n_imgs, shape=(4, 5, 6), rng=None):
    if rng is None:
        rng = _cluster_rng()
    return [Nifti1Image(rng.randn(*shape), np.eye(4))
            for _ in range(n_imgs)]


def _reference_cr2_variance(Y, X, groups, con_val):
    """Direct, unvectorized CR2 cluster sandwich for a single contrast."""
    xtx_inv = np.linalg.inv(np.dot(X.T, X))
    beta = np.dot(xtx_inv, np.dot(X.T, Y))
    residuals = Y - np.dot(X, beta)
    meat = np.zeros((X.shape[1], X.shape[1], Y.shape[1]))
    for group in np.unique(groups):
        indices = np.where(groups == group)[0]
        x_g = X[indices]
        hat_g = np.dot(x_g, np.dot(xtx_inv, x_g.T))
        adjustment_matrix = np.eye(indices.size) - hat_g
        adjustment_matrix = (adjustment_matrix + adjustment_matrix.T) / 2.
        eigenvalues, eigenvectors = np.linalg.eigh(adjustment_matrix)
        eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).eps)
        adjustment = np.dot(
            eigenvectors,
            np.dot(np.diag(1. / np.sqrt(eigenvalues)), eigenvectors.T))
        score = np.dot(x_g.T, np.dot(adjustment, residuals[indices]))
        meat += score[:, None, :] * score[None, :, :]
    bread = xtx_inv
    variance = np.einsum('i,ijk,j->k',
                         np.dot(con_val, bread), meat,
                         np.dot(bread, con_val))
    return beta, variance


def test_cluster_robust_variance_matches_reference_cr2():
    rng = _cluster_rng()
    n_observations, n_regressors, n_voxels = 8, 2, 7
    X = np.column_stack([np.ones(n_observations),
                         rng.randn(n_observations)])
    Y = rng.randn(n_observations, n_voxels)
    groups = np.array(['s1', 's1', 's2', 's2', 's3', 's3', 's4', 's4'])
    con_val = np.array([0., 1.])

    cluster_info, xtx_inv = _cluster_design_quantities(X, groups)
    beta = OLSModel(X).fit(Y).theta
    variance = _cluster_robust_contrast_variance(
        Y, X, beta, con_val, cluster_info, xtx_inv)

    beta_ref, variance_ref = _reference_cr2_variance(
        Y, X, groups, con_val)
    assert_almost_equal(beta, beta_ref)
    assert_almost_equal(variance, variance_ref)


def test_cluster_robust_singleton_groups_match_hc2():
    """CR2 collapses to the HC2 sandwich when every cluster is a singleton.
    """
    rng = _cluster_rng()
    n_observations, n_regressors, n_voxels = 6, 2, 4
    X = np.column_stack([np.ones(n_observations),
                         rng.randn(n_observations)])
    Y = rng.randn(n_observations, n_voxels)
    groups = np.array(['s%i' % i for i in range(n_observations)])
    con_val = np.array([1., 0.])

    cluster_info, xtx_inv = _cluster_design_quantities(X, groups)
    beta = OLSModel(X).fit(Y).theta
    variance = _cluster_robust_contrast_variance(
        Y, X, beta, con_val, cluster_info, xtx_inv)

    hat_diag = np.diag(np.dot(X, np.dot(xtx_inv, X.T)))
    residuals = Y - np.dot(X, beta)
    weights = np.dot(con_val, np.dot(xtx_inv, X.T)) ** 2
    hc2_variance = np.sum(
        weights[:, None] * residuals ** 2 / (1. - hat_diag)[:, None],
        axis=0)
    assert_almost_equal(variance, hc2_variance)


def test_second_level_cluster_model_t_contrast():
    rng = _cluster_rng()
    n_observations = 8
    effect_maps = _make_effect_imgs(n_observations, rng=rng)
    design_matrix = pd.DataFrame(
        {'intercept': np.ones(n_observations),
         'covariate': rng.randn(n_observations)})
    groups = ['s1', 's1', 's2', 's2', 's3', 's3', 's4', 's4']
    mask = Nifti1Image(np.ones((4, 5, 6), np.int8), np.eye(4))

    model = SecondLevelModel(
        mask_img=mask, noise_model='cluster').fit(
        effect_maps, design_matrix=design_matrix, groups=groups)

    assert model.n_groups_ == 4
    # declared independent-group rule: G - 1 residual degrees of freedom
    assert model.df_residuals_ == 3

    outputs = model.compute_contrast('covariate', output_type='all')
    stat_map = outputs['stat']
    assert isinstance(stat_map, Nifti1Image)
    stat_data = get_data(stat_map)
    assert np.isfinite(stat_data).all()

    # the t statistics must be based on the CR2 variance and G - 1 dof
    Y = model.masker_.transform(effect_maps)
    X = design_matrix.values
    beta_ref, variance_ref = _reference_cr2_variance(
        Y, X, np.asarray(groups), np.array([0., 1.]))
    effect_ref = beta_ref[1]
    t_ref = effect_ref / np.sqrt(variance_ref)
    assert_almost_equal(stat_data[get_data(mask) > 0], t_ref)
    assert_almost_equal(
        get_data(outputs['effect_variance'])[get_data(mask) > 0],
        variance_ref)

    # p-values are Student t with G - 1 = 3 dof, not normal and not n - p
    from scipy.stats import t as t_distribution
    p_ref = t_distribution.sf(t_ref, 3)
    assert_almost_equal(
        get_data(outputs['p_value'])[get_data(mask) > 0], p_ref)


def test_cluster_model_keeps_ols_point_estimates():
    rng = _cluster_rng()
    effect_maps = _make_effect_imgs(6, rng=rng)
    design_matrix = pd.DataFrame({'intercept': np.ones(6)})
    groups = ['s1', 's1', 's2', 's2', 's3', 's3']
    mask = Nifti1Image(np.ones((4, 5, 6), np.int8), np.eye(4))

    ols_model = SecondLevelModel(mask_img=mask).fit(
        effect_maps, design_matrix=design_matrix)
    cluster_model = SecondLevelModel(
        mask_img=mask, noise_model='cluster').fit(
        effect_maps, design_matrix=design_matrix, groups=groups)

    ols_effect = get_data(
        ols_model.compute_contrast(output_type='effect_size'))
    cluster_effect = get_data(
        cluster_model.compute_contrast(output_type='effect_size'))
    assert_almost_equal(ols_effect, cluster_effect)

    # but the standard errors and dof differ: repeated rows are not
    # independent observations
    ols_variance = get_data(
        ols_model.compute_contrast(output_type='effect_variance'))
    cluster_variance = get_data(
        cluster_model.compute_contrast(output_type='effect_variance'))
    assert not np.allclose(ols_variance, cluster_variance)
    assert ols_model.results_[0.0].df_residuals == 5
    assert cluster_model.df_residuals_ == 2


def test_cluster_model_rejects_bad_inputs():
    rng = _cluster_rng()
    effect_maps = _make_effect_imgs(6, rng=rng)
    design_matrix = pd.DataFrame({'intercept': np.ones(6)})

    # unknown noise model
    with pytest.raises(ValueError):
        SecondLevelModel(noise_model='robust')

    cluster_model = SecondLevelModel(noise_model='cluster')

    # plain niimg input without explicit groups
    with pytest.raises(ValueError, match='group labels'):
        cluster_model.fit(effect_maps, design_matrix=design_matrix)

    # a single group cannot estimate any between-subject variance
    with pytest.raises(ValueError, match='at least two'):
        cluster_model.fit(
            effect_maps, design_matrix=design_matrix,
            groups=['s1'] * 6)

    # group labels must be aligned with the image rows
    with pytest.raises(ValueError, match='one label per image row'):
        cluster_model.fit(
            effect_maps, design_matrix=design_matrix,
            groups=['s1', 's2'])

    # rank deficient design: fixed effects not estimable
    singular_design = pd.DataFrame(
        {'intercept': np.ones(6), 'duplicate': np.ones(6)})
    with pytest.raises(ValueError, match='not full rank'):
        cluster_model.fit(
            effect_maps, design_matrix=singular_design,
            groups=['s1', 's1', 's2', 's2', 's3', 's3'])

    # F contrasts are not supported by the cluster-robust covariance
    fitted = SecondLevelModel(noise_model='cluster').fit(
        effect_maps, design_matrix=design_matrix,
        groups=['s1', 's1', 's2', 's2', 's3', 's3'])
    with pytest.raises(ValueError, match='F contrasts'):
        fitted.compute_contrast(second_level_stat_type='F')


def test_cluster_model_infers_groups_from_dataframe():
    rng = _cluster_rng()
    with InTemporaryDirectory():
        paths = []
        for i in range(6):
            img = Nifti1Image(rng.randn(3, 3, 3), np.eye(4))
            path = 'map_%02i.nii.gz' % i
            save(img, path)
            paths.append(path)
        dataframe = pd.DataFrame({
            'subject_label': ['s1', 's1', 's2', 's2', 's3', 's3'],
            'map_name': ['con'] * 6,
            'effects_map_path': paths})
        model = SecondLevelModel(noise_model='cluster').fit(dataframe)
        assert model.n_groups_ == 3
        assert model.df_residuals_ == 2
        assert_array_equal(
            model.groups_,
            np.array(['s1', 's1', 's2', 's2', 's3', 's3']))
        stat_map = model.compute_contrast(
            first_level_contrast='con', output_type='stat')
        assert isinstance(stat_map, Nifti1Image)
        assert np.isfinite(get_data(stat_map)).all()


def test_cluster_model_repeated_measures_use_subject_dof():
    rng = _cluster_rng()
    subject_means = rng.randn(4, 3)
    # two noisy repeats per subject
    data = np.vstack([subject_means + 0.01 * rng.randn(4, 3),
                      subject_means + 0.01 * rng.randn(4, 3)])
    groups = ['s1', 's2', 's3', 's4'] * 2
    effect_maps = [
        Nifti1Image(data[i].reshape(1, 1, 3), np.eye(4))
        for i in range(8)]
    design_matrix = pd.DataFrame({'intercept': np.ones(8)})
    mask = Nifti1Image(np.ones((1, 1, 3), np.int8), np.eye(4))

    cluster_model = SecondLevelModel(
        mask_img=mask, noise_model='cluster').fit(
        effect_maps, design_matrix=design_matrix, groups=groups)
    assert cluster_model.df_residuals_ == 3

    outputs = cluster_model.compute_contrast(output_type='all')
    Y = cluster_model.masker_.transform(effect_maps)
    X = design_matrix.values
    beta_ref, variance_ref = _reference_cr2_variance(
        Y, X, np.asarray(groups), np.array([1.]))
    assert_almost_equal(
        get_data(outputs['effect_variance']).ravel(), variance_ref)

    # naive OLS treats the repeats as independent: more (wrong) dof and
    # anti-conservative variance under intra-subject correlation
    ols_model = SecondLevelModel(mask_img=mask).fit(
        effect_maps, design_matrix=design_matrix)
    ols_variance = get_data(
        ols_model.compute_contrast(
            output_type='effect_variance')).ravel()
    assert ols_model.results_[0.0].df_residuals == 7
    assert np.all(ols_variance < variance_ref)
