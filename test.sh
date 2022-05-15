for i in {1..9}
do
	python snackysnek.py --test="10"
	python snackysnek.py --pathfinder="astar" --test="10"
done
PAUSE