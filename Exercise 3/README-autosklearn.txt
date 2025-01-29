For Starting auto-sklearn Docker in Windows or Mac
-----------------------------------
 To download from dockerhub, use:  docker pull mfeurer/auto-sklearn:master   (https://automl.github.io/auto-sklearn/master/installation.html)
 To get docker id use: docker ps
 To copy files to docker use: docker cp AutoSklearn.py <container_id>:/AutoSklearn.py
			      docker cp airfoil_noise_data.csv <container_id>:/airfoil_noise_data.csv
			      docker cp CongressionalVotingID.shuf.lrn.csv <container_id>:CongressionalVotingID.shuf.lrn.csv
		              docker cp abalone.csv <container_id>:/abalone.csv

Install python in docker:  docker run -it mfeurer/auto-sklearn:master
                           pip install python3
To run the python script:
	docker exec -it <container_id> bash
	python3 /AutoSklearn.py		