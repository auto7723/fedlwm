import torch

from fedlwm.geometry import HypersphericalReferenceFrame


def test_reference_frame_is_fixed_and_tangent():
    frame = HypersphericalReferenceFrame(7, 16, 4)
    assert not frame.anchors.requires_grad
    assert not frame.bases.requires_grad
    for cls in range(7):
        basis = frame.bases[cls]
        torch.testing.assert_close(basis.T @ basis, torch.eye(4), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(frame.anchors[cls] @ basis, torch.zeros(4), atol=1e-5, rtol=1e-5)
    gram = frame.anchors @ frame.anchors.T
    off_diagonal = gram[~torch.eye(7, dtype=torch.bool)]
    torch.testing.assert_close(off_diagonal, torch.full_like(off_diagonal, -1.0 / 6.0), atol=1e-5, rtol=1e-5)


def test_log_map_has_requested_dimension():
    frame = HypersphericalReferenceFrame(3, 8, 2)
    value = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    result = frame.log_map_all(value)
    assert result.shape == (5, 3, 2)
