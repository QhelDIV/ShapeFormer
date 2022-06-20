sudo docker run --gpus all -it \
--name=shapeformer \
-p 4949:22   \
-p 4950:5901 \
-p 4951:3389 \
-p 4952:443  \
-p 4953:80   \
-p 4954:8888 \
-p 4955:6006 \
--ipc=host   \
-v /studio:/studio \ # REPLACE the /studio to your desired physical host location, for example ~/home/workshop/
shapeformer:latest \
bash

#-v qinglong:/studio \

#pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
#--privileged \
#--cap-add SYS_ADMIN \
# allow using vpn (openconect)
#--cap-add NET_ADMIN --device /dev/net/tun myimage
#--cap-add DAC_READ_SEARCH \
