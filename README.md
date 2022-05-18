# SnakePlayer
### AI Snake game player

##### Quickstart:
`python snackysnek.py`

You can add your own parameters:
- `--pathfinder` (basic/astar/hamiltonian), default is basic
- `--game_speed` (the higher the number, the faster the game), default is 200

#### Dependencies:
- numpy
- pygame

##### Files:
- `snackysnek.py` --> The snake game
- `pathfinding_algorithm.py` --> Path-finding algorithm
- `testing.py` --> tests the performance of the algorithms
- `test.sh` --> runs game 300 times (100 times per algorithm)


##### Game rules:
- Everytime snake reaches the apple it gets longer for 1 square
- Snake dies if it crashes into itself or into the wall
- Snake wins if it reaches full length
