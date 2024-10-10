  # prepare system
  sudo apt update -y
  sudo apt upgrade -y
  sudo apt install python3.10-venv
  sudo apt install build-essential
  sudo apt install nvidia-driver-525 nvidia-dkms-525
  sudo apt install nvidia-cuda-toolkit

  # prepare python env
  python3 -m venv env
  source env/bin/activate
  
  # install weights
  mkdir weights
  cd weights
  wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  
  # install gcc 10
  sudo apt install gcc-10 g++-10
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 20
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 20
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
  export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-10'

  # install grounding dino
  git clone https://github.com/IDEA-Research/GroundingDINO.git
  cd GroundingDINO
  pip install -q -e .

  # install sam
  pip install git+https://github.com/facebookresearch/segment-anything.git

  # install requirements
  pip install -r requirements.txt

  # install systemd services
  sudo cp fastapi.service /etc/systemd/system/
  sudo systemctl enable fastapi.service
  sudo systemctl restart fastapi.service

  # install nginx
  sudo apt install nginx
  cd /etc/nginx/sites-enabled/
  sudo cp /home/ubuntu/fastapi_nginx .
  sudo service nginx restart
