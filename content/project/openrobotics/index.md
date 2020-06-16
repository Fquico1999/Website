---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "UBC Open Robotics"
summary: "Developing Software to compete in the RoboCup@Home competition."
authors: []
tags: ["Deep Learning", "Computer Vision"]
categories: []
date: 2020-06-05T15:18:22-07:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: true

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

_*[Click here](#contrib) if you want to skip to my involvement in this team.*_

## Overview
[UBC Open Robotics](http://www.ubcopenrobotics.ca/) is a student team comprised of 60 students split into three subteams - ArtBot, PianoBot, and Robocup@Home. I am a member of the software team in the RoboCup@Home subteam.

The objective of RoboCup@Home is to build a household assistant robot that can perform a variety of tasks, including carrying bags, introducing and seating guests at a party, answering a variety of trivia questions and more. Open Robotics is developing a robot to compete in the 2021 [RoboCup@Home Education Challenge](https://www.robocupathomeedu.org/) while in the meantime, our subteam will compete in the 2020 Competition using the Turtlebot 2 as our hardware platform.

## The Challenge
<a name="task1"></a>
The rules for the 2020 Challenge can be found [here](https://docs.google.com/document/d/1aNPdZDvf9X4HHF13eSge_eHDP9NmC6UDqeYDM9Xyjcg/edit), but they boil down to three specific tasks:
- Carry My Luggage - Navigation task
- Find My Mates - Vision task
- Receptionist - Speech task

#### Carry My Luggage

*Goal: The robot helps the operator to carry a bag to the car parked outside*

Starting at a predifined location, the robot has to find the operator and pick up the bag the operator is pointing to. After picking up the bag, the robot needs to indicate that it is ready to follow and then it must follow the operator while facing 4 obstacles along the way (crowd, small object, difficult to see 3D object, small blocked area).

#### Find My Mates

*Goal: The robot fetches the information of the party guests for the operator who knows only the names
of the guests.*
<a name="task3"></a> 
Knowing only the operator, the robot must identify unknown people and meet those that are waving. Afterwards, it must remember the person and provide a unique description of that person, as well as that person's location, to the operator.

#### Receptionist

*Goal: The robot has to take two arriving guests to the living room, introducing them to each other,
and offering the just-arrived guest an unoccupied place to sit.*
<a name="contrib"></a>
Knowing the host of the party, John, the robot must identify unknown guests, request their names and favourite drinks and then point to an empty seat where the guest can sit.

## My Contributions

My main contributions have been in speech recognition and in handle segmentation, targeting [task 3](#task3) and [task 1](#task1) respectively, however I also worked on facial recognition earlier in the project.

### Speech Recognition
*You can find this repository [here](https://github.com/UBC-OpenRobotics/SpeechRecognition)*

### Handle Segmentation
*You can find this repository [here](https://github.com/UBC-OpenRobotics/HandleSegmentation)*

{{< video library="1" src="openrobotics_handleseg.mp4" controls="" >}}