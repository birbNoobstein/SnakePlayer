# SnakePlayer
### AI that plays Snake using A* and reverse A* algorithm (probably not sure yet ?)

#### Dependencies:
- matplotlib
- numpy
- pyautogui
- selenium
- webdriver_manager

(or import from `environment.yml` file)

#### Initiation
Webdriver manager is used to install Chrome driver for selenium webdriver. Selenium opens [Snake game online](https://www.coolmathgames.com/0-snake/play). Further on pyautogui is used to manage things.<br>
In order to start the game, pyautogui presses space and the initial game frame is shown. The screenshot of whole site is taken in order to locate the game-window.<br>
The game-window consists of 20 rows and X columns. The game-window dimentions are used to compute the single square side size and 2D-array is created to represent the play-grid. Play-window width and square site size are then used to compute X (number of columns). Each number in the array represents the middle of each square in the play-grid and each pixel in the game-window screenshot is checked for it's color.<br><br>
![](https://github.com/birbNoobstein/SnakePlayer/blob/main/plgrnd.png)
| Pixel color        | Represents        | Array value  |
| :----------------: |:-----------------:| :-----------:|
| Black              | Background        |    0         |
| Red                | Apple             |    1         |
| Green              | Snake             |    2         |


