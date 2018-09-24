Tensorflow with Docker


docker run -it -p 8888:8888 tensorflow/tensorflow
docker run -it --rm --name tf -v ~/mywork/notebooks -p 8888:8888 -p 6006:6006 tensorflow/tensorflow
docker exec -it tf tensorboard --logdir tf_logs/


To clean docker images:

docker rmi $(docker images --filter "dangling=true" -q --no-trunc)


https://www.youtube.com/watch?v=tYYVSEHq-io&t=6504s

What's your ML Test Score? A   rubric for ML production system

https://www.eecs.tufts.edu/~dsculley/papers/ml_test_score.pdf


ML Google crash course

https://developers.google.com/machine-learning/crash-course/ml-intro


HOW TO RUN JUPITER, KERAS, TENSORFLOW, PANDAS, SKLEARN AND MATPLOTLIB IN DOCKER CONTAINER
https://dev-ops-notes.com/docker/howto-run-jupiter-keras-tensorflow-pandas-sklearn-and-matplotlib-docker-container/