# The Ant Farm

The Ant Farm is a program that aims to simplify the process of PCB-making using CNC machines.  
In this software we included both the necessary CAM features and the CNC machine control capabilities, trying to simplify the user operations as much as possible.  

# Actual status  
This software is a pre-alpha, meaning that there are no warranties and safety of the software and its use.
Up to now, the application is compatible only with CNC machines using GRBL v1.1 firmware, and it has been tested using only some gerber and drill files generated using Autodesk's Eagle and KiCad EDA.  
If you want to have some more information about the actual implemented features, you could watch this video:  
  
[![Software features video](https://img.youtube.com/vi/3Gy312kk_yw/0.jpg)](https://www.youtube.com/watch?v=3Gy312kk_yw)  

# Getting started  
  
## Prerequisites  

The application requires an installation of python version 3.7 or greater.  

## Installation  
  
The python packages needed are listed in the file "requirements.txt", that could be used to install them by command line.  
The cleanest way to install these packages is using a virtual environment.

### Linux  

Either download the zip of the repository sources or use git:  
  
> git clone https://github.com/TheAntTeam/TheAntFarm.git  

Enter the folder where there is the code, create a virtual environment and activate it:  
  
> cd TheAntFarm  
> python3 -m venv ./env  
  
Activate the virtual environment:  
    
> source ./env/bin/activate

Install all the required packages:
  
> pip3 install -r requirements.txt  

It could be also necessary to install an additional library:
  
> sudo apt install libxcb-xinerama0  
  
  
  
### Windows  

Either download the zip of the repository sources or use git:  
  
> git clone https://github.com/TheAntTeam/TheAntFarm.git  

Enter the folder where there is the code, create a virtual environment and activate it:  
  
> cd TheAntFarm    
> python3 -m venv .\env  
  
Activate the virtual environment:  
  
> .\env\Scripts\activate  
  
Install all the required packages:
  
> pip3 install -r requirements.txt  

## Running the software  
  
The software must be run from the project directory, after the virtual environment has been activated (see installation paragraph), using the following command:

> python3 the_ant_farm.py  
  
    
# Disclaimer  
  
The providers of this software decline any responsibility for damages to persons or things deriving from its use, and they will not be liable for any damages you may suffer in connection with using, modifying, or distributing this SOFTWARE PRODUCT.  


# Donation

Our projects requires a lot of work and often expensive hardware for testing (CNC machines).  
Please consider a safe, secure and highly appreciated donation via the PayPal link below.  
  
  
[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=BTRCVPZUZYW2E)  
  
