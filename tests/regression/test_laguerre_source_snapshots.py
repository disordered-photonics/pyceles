from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.fields import (
    FocusedLaguerreGaussianBeam,
    LaguerreGaussianBeam,
    project_source_to_svwf,
)

pytestmark = pytest.mark.reference


def test_laguerre_gaussian_projection_snapshot():
    source = LaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=1,
        azimuthal_order_l=2,
        polarization=(1.0 + 0.0j, -0.3 + 0.4j),
        polar_angle=0.25,
        azimuthal_angle=0.55,
        beam_width=1700.0,
        focal_point=(10.0, -5.0, 2.0),
        amplitude=1.1,
        azimuthal_phase=0.15,
    )
    coeffs = project_source_to_svwf(
        np.array([[0.0, 0.0, 0.0], [80.0, -20.0, 30.0]], dtype=float),
        3,
        source,
        polar_angles=np.linspace(0.0, np.pi, 241),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 161, endpoint=False),
    )
    expected0 = np.array(
        [
            0.0023693818120903 - 0.0010405891500360j,
            0.0023916922827709 - 0.0290310822586665j,
            0.0551018434989179 + 0.0630977574023025j,
            0.0008510880505994 + 0.0024519231922324j,
            0.0178242436287991 + 0.0129871549448046j,
            0.1256122460809809 - 0.0060291598797695j,
            -0.2264507253840601 + 0.2008810991090584j,
            -0.0225175022786088 + 0.0648428122161460j,
            -0.0011502431700532 - 0.0001647484109299j,
            -0.0119801662438680 + 0.0058807621457649j,
            -0.0415943199409208 + 0.0738122457392579j,
            0.0367375362905412 + 0.3448129352069269j,
        ],
        dtype=np.complex128,
    )
    expected1 = np.array(
        [
            0.0011731014742988 + 0.0163742176971597j,
            -0.0742253559547149 + 0.0287206791005006j,
            0.0053798245709163 + 0.1071808026032386j,
            -0.0028600228211387 - 0.0031098135235161j,
            -0.0095941135196851 - 0.0284664194926005j,
            0.0017541882550323 - 0.1315006408800990j,
            -0.3430824287288113 + 0.0300815411339878j,
            -0.1225508577262907 + 0.1070164012286689j,
            0.0011221261786287 + 0.0003050241869210j,
            0.0098172921035875 + 0.0037422045330475j,
            0.0575283403095997 + 0.0250076445615193j,
            0.1433712697491123 + 0.1512361348069911j,
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(coeffs[0, :12], expected0, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(coeffs[1, :12], expected1, rtol=1e-11, atol=1e-11)


def test_focused_laguerre_projection_snapshot():
    source = FocusedLaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=1,
        polarization=(1.0 + 0.0j, 0.2j),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1200.0,
        focal_length=1000.0,
        numerical_aperture=0.75,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
        azimuthal_phase=0.0,
        sine_condition_apodization=True,
    )
    coeffs = project_source_to_svwf(
        np.array([[0.0, 0.0, 0.0], [45.0, -30.0, 15.0]], dtype=float),
        3,
        source,
        polar_angles=np.linspace(0.0, np.pi, 301),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False),
    )
    expected0 = np.array(
        [
            1.0039712117215771e-16 + 2.1857515797307769e-16j,
            2.7321894746634712e-17 + 7.3813282075226228e-01j,
            -7.3237856751395825e-17 - 6.1582683397176652e-17j,
            5.3776427755281020e-17 + 1.4203048459560108e-17j,
            4.6837533851373792e-17 - 8.5719734262135194e-17j,
            -1.3069225077434259e00 - 1.3498317047444530e-17j,
            0.0000000000000000e00 - 7.6056782399858136e-17j,
            8.0491258232658924e-01 + 1.1641620827063104e-17j,
            -3.2526065174565133e-19 - 4.8572257327350599e-17j,
            9.3024546399256280e-17 - 8.1532003370909933e-17j,
            -1.0955862952966022e-16 + 1.7076184216646695e-18j,
            -2.5695591487906455e-17 - 1.4975489229506720e00j,
        ],
        dtype=np.complex128,
    )
    expected1 = np.array(
        [
            2.5547862003675020e-01 - 1.2405853106336792e-01j,
            -1.0385325958062304e-01 + 7.0169553272510221e-01j,
            1.4555344510075324e-01 - 1.0425386050107553e-01j,
            -1.0273608778125477e-02 + 1.7713829824653794e-02j,
            9.4440642054588908e-02 + 1.9570927836766017e-01j,
            -1.2441028088984059e00 - 1.8546179432531795e-01j,
            1.3309485385246275e-01 + 4.9976559902256940e-02j,
            7.7093914171050604e-01 + 1.0847437852538799e-01j,
            -6.5179222127859354e-05 - 9.9163220303387844e-04j,
            -2.1437248463860976e-02 - 1.2477490263732509e-02j,
            -6.4161616988005210e-02 + 2.9934606660626303e-02j,
            2.1604172708614419e-01 - 1.4293641518501392e00j,
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(coeffs[0, :12], expected0, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(coeffs[1, :12], expected1, rtol=1e-11, atol=1e-11)
