For Starting auto-sklearn Docker in Windows or Mac
-----------------------------------
 To download from dockerhub, use:  docker pull mfeurer/auto-sklearn:master
 To get docker id use: docker ps
 To copy files to docker use: docker cp AutoSklearn.py <container_id>:/AutoSklearn.py
			      docker cp airfoil_noise_data.csv <container_id>:/airfoil_noise_data.csv
		              docker cp abalone.csv <container_id>:/abalone.csv

To run the python script:
	docker exec -it <container_id> bash
	python3 /AutoSklearn.py		