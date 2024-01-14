import argparse

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from skimage import exposure

from dataset import ActiveFire
from trainer import UNetTrainer


def stretch_image(im: np.ndarray) -> np.ndarray:
    """Stretches the contrast of an image by adjusting the instensities
    based on the 1st and 99th percentiles.

    Args:
        im (np.ndarray): input image to be strecthed

    Returns:
        np.ndarray: output image

    """
    p1, p99 = np.percentile(im, (1, 99))
    J = exposure.rescale_intensity(im, in_range=(p1, p99))
    J = J / J.max()
    return J


def main():
    """Run inference on the test set and save the results."""

    parser = argparse.ArgumentParser(description="Active Fire Detection")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=r"./fire_unetsimple/04_activefire_512epoch=417-val_loss=0.09.ckpt",  # noqa: E501
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--fire_path",
        type=str,
        default=r"/data/active_fire_dataset/fire_images",
        help="Directory path for fire images",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=r"/data/active_fire_dataset/fire_masks",
        help="Directory path for fire masks",
    )
    parser.add_argument(
        "--input-ch", default=10, type=int, help="Number of input channels"
    )
    parser.add_argument(
        "--encoder-ch",
        default=(32, 64, 128, 256, 512, 1024, 2048),
        type=int,
        help="Encoder channels",
    )
    parser.add_argument(
        "--use-act", default=True, type=int, help="Encoder channels"
    )  # noqa: E501
    parser.add_argument("--lr", default=1e-4, type=int, help="Learning rate")
    parser.add_argument(
        "--tb-log-pred-gt",
        default=False,
        type=int,
        help="Viz pred and gt with TB",  # noqa: E501
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    print(checkpoint.keys())

    model = UNetTrainer(
        input_ch=args.input_ch,
        enc_ch=args.encoder_ch,
        use_act=args.use_act,
        lr=args.lr,
        tb_log_pred_gt=args.tb_log_pred_gt,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    test_dataset = ActiveFire(
        img_dir=args.fire_path, mask_dir=args.mask_path, split="test"
    )

    for idx in tqdm.tqdm(range(len(test_dataset))):
        image, mask, id = test_dataset[idx]
        image = torch.Tensor(image).to(device)

    logits = model(image.unsqueeze(0))

    image_viz = stretch_image(
        np.transpose(image.detach().cpu().numpy(), (1, 2, 0))[:, :, [3, 2, 1]]
    )
    predc = logits.sigmoid()
    f = plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_viz)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze())
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predc.detach().cpu().numpy().squeeze())
    plt.title("Prediction")
    plt.axis("off")

    plt.savefig(f"predictions/pred_{id}")

    plt.show()

    f.clear()
    plt.close(f)

    del image, logits, mask, predc


if __name__ == "__main__":
    main()
