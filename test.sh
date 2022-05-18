for i in {1..10}
do
	python snackysnek.py --test="10" --game_speed="10000"
	python snackysnek.py --pathfinder="astar" --test="10" --game_speed="10000"
	python snackysnek.py --test="10" --pathfinder="hamiltonian" --game_speed="10000"
done

PAUSE