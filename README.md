# DroneReinforcementLearning

Here are the steps to compile and run our code in order to replicate our results:

1.) Go to this link: https://drive.google.com/file/d/1sa3BACGzJf0FKjR2DPEz-j345Ub-pG0F/view?usp=sharing

2.) Run all of the cells top to bottom

3.) Click on the folder icon on the left side of the screen 

4.) If you don't see a file called "PPO.zip" then click the refresh icon and then it should popup. Download this file.

5.) Go to this website: https://microsoft.github.io/AirSim/build_macos/ and follow the instructions to install AirSim on your computer. They have instructions for Windows, Mac, and Linux.

6.) Open AirSim.

7.) Click "Play"

8.) When the window pops up asking "Would you like to use car simulation? Choose no to use quadrotor simulation." select No.

9.) You should now see a drone. Now open a terminal on your computer.

10.) Download the "RubenTest.py" script from this repository and save it somewhere on your computer.

11.) Now, using Terminal, navigate to the folder where RubenTest.py is saved.

12.) Upload "PPO.zip" to this folder.

13.) Type "python RubenTest.py" and press enter.

14.) You will see it say "Press any key to takeoff" in the terminal, press any key and then follow the rest of the prompts. You should see the drone flying autonomously in AirSim using the reinforcement learning model that you trained (PPO.zip).
