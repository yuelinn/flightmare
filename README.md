# DodgeDrone Challenge: 

Hello, thanks for taking the time to look at our submission for the DodgeDrone Challenge. We are team LhylC comprising of Huiyu Leong and Yue Linn Chong. We are open to suggestions and feel free to contact us either by submitting an issue or email me at yuelinnchong@gmail.com :)


## Instructions

We assume that you have already had docker installed. We tested the code with docker version 20.10.6, build 370c289:
To run our submission, please  follow the instructions below:
1. Pull our docker image:   
    `docker pull linncy/dodgedrone-challenge:submission-Team-LhylC`   
   This repo should already be inside the docker at the path /root/flightmare. The dependencies should also be already installed. 
   BTW, we don't use conda (its not personal, we just decided we don't need to for this project) so there is nothing to activate.
2. Run the container:     
    `docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all  -p 10253:10253 -p 10254:10254 --name ddc-challenge uzhrpg/dodgedrone-challenge:base /bin/bash`   
3. In a different terminal (not in the docker), run the Unity standalone. (The competition/Flightmare provided the binaries that can be downloaded [here](http://rpg.ifi.uzh.ch/challenges/DodgeDrone2021/Standalone_Forest.zip))   
  `./STANDALONE_PATH/RPG_Flightmare.x86_64 -static-obstacles 0`
4. To evaluate our submission, run within the docker:    
   `cd $FLIGHTMARE_PATH/flightrl/rpg_baselines/evaluation`   
   `python3 evaluation.py --policy_path $POLICY_PATH`   
   As of May 8 2044h GMT+8, POLICY_PATH=/root/fligtmare/flightrl/rpg_baselines/evaluation/waypt_policy


## Publication

Since this is for the DodgeDrone Challenge, the code in this repo is heavily based on the following paper **[PDF](http://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf)** (citation below) and its code (parent of this repo).

```
@article{song2020flightmare,
    title={Flightmare: A Flexible Quadrotor Simulator},
    author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
    booktitle={Conference on Robot Learning},
    year={2020}
}
```
