# DroneReinforcementLearning

IMPORTANT NOTE: automatically generated documentation for the code is located at doc > _build > html > index.html

Software you must install in order for the steps below to work:

1.) You need to do pip install gym

2.) You need to do pip install stable-baselines3[extra] (if you are on Mac then you need to do pip install 'stable-baselines3[extra]')


Here are the steps to compile and run our code in order to replicate our results:

1.) Go to this link: https://colab.research.google.com/drive/1-P9xrSE-l14V0mS0qsHiY_ZmHGxPqx01?usp=sharing

2.) Run all of the cells top to bottom

3.) Once all of the cells have finished running, click on the folder icon on the left side of the screen 

4.) If you don't see a file called "PPO.zip" then click the refresh icon and then it should popup. Download this file.

5.) Go to this website: https://microsoft.github.io/AirSim/build_macos/ and follow the instructions to install AirSim on your computer. Follow the "Host machine" instructions. They have instructions for Windows, Mac, and Linux.

6.) Open AirSim (follow the instructions here to open AirSim: https://microsoft.github.io/AirSim/build_macos/. Look at the section titled "How to Use AirSim").

7.) Click the dropdown with the double arrow pointing to the right next to where it says "Compile" (see image below). Then Click "Play".

![Cover Photo](/images/dropdown.png)

8.) When the window pops up asking "Would you like to use car simulation? Choose no to use quadrotor simulation." select No.

9.) You should now see a drone.

10.) Download the "AirSimRL.py" script from this GitHub repository and save it in the following directory location: /AirSim/PythonClient/multirotor

11.) Upload "PPO.zip" to the same folder (/AirSim/PythonClient/multirotor)

12.) Open up a Terminal.

13.) Now, using Terminal, navigate to /AirSim/PythonClient/multirotor.

14.) Type "python AirSimRL.py" and press enter.

15.) You will see it say "Press any key to takeoff" in the terminal, press any key and then follow the rest of the prompts. You should see the drone flying autonomously in AirSim using the reinforcement learning model that you trained (PPO.zip).

Now the below steps pertain to our second project:

16.) Go to the following link " https://colab.research.google.com/drive/1u7PHpb2CAS7DMYpy9sBmvOhdYCzVwZuC?usp=sharing "

17.) Run all of the cells top to bottom

18.) Once all of the cells have finished running, below the last cell you will see the output being displayed. 

19.) You will be seeing two matrices, the first one gives you the general matrix like rows and numbers , where the obstacles are located, goal and start position.

20.) The second bottom matrix gives you the reward values updated in all the cells. and you can notice the reward values neat the obstacle are lower comapared to the other cells/states.

-----------------------------------------------------------------------------------
Information about datasets:

No datasets were used as part of this preliminary project code since we utilized reinforcement learning.
