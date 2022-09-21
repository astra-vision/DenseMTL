# Adapted from work by Hoyer et al.
# https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth

import os, hashlib, urllib, zipfile

from .google_drive_downloader import GoogleDriveDownloader


def download_model(model_name, download_dir=None):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = dict(
        mono_cityscapes_1024x512_r101dil_aspp_dec5=
            ("gdrive_id=1VF86Wqv9x7afLt_B8t2OaWtb-lG0vwyN", ""),
        mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2=
            ("gdrive_id=1Kki3vwDxCeSdLQI5LLJVwk7erTk6EVkB", ""),
        mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5=
            ("gdrive_id=19rJIafDLyAW348bYE3M_EoQcIK0OIj0V", ""),
        mono_cityscapes_1024x512_r101dil_aspp_dec5_posepretrain_crop512x512bs4=
            ("gdrive_id=1V3qzmCIfErOhLILnwCCchYMkaKLtUA7c", ""),
        mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4=
            ("gdrive_id=1woRzEPVuhaafrS_2_GlsJuVRyxWaGO4O", ""),
        mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd0_crop512x512bs4=
            ("gdrive_id=1G7bDZ-0PsHeMSHK59EqJn5ncqMzWB1Js", ""),
        mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs2=
            ("gdrive_id=1bHlAYHKSv6sVbQBMlQ-D7kkUcAMb8-Jq", ""))

    if download_dir is None:
        download_dir = os.environ['DOWNLOAD_DIR']
        download_dir = os.path.expandvars(download_dir)
        download_dir = download_dir.replace('$SLURM_JOB_ID/', '')
    os.makedirs(download_dir, exist_ok=True)
    model_path = os.path.join(download_dir, model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
            print('Monodepth2 model download checksum', current_md5checksum)
        return True
        # return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "depth.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            if "https://" in model_url:
                urllib.request.urlretrieve(model_url, model_path + ".zip")
            else:
                model_url = model_url.replace("gdrive_id=", "")
                GoogleDriveDownloader.download_file_from_google_drive(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))