**Highlevel ML Project WorkFlow with MLOps**

1. Create Github Repo of the Project using Template
2. Clone repo on local system
3. Initialize dvc
    + dvc init
    + git add .dvc .gitignore
    + git commit -m "Initialize DVC"
    + dvc remote add -d prashant-mlops-bucket s3://prashant-mlops-bucket
4. upload data on s3
5. perform experiments and find best algorithm & parameters - set everything on mlflow
6. create dvc pipeline, save code on github, data on s3 via dvc, model on model registry
7. Apply CI: run dvc pipeline, perform tests ----> promote model in mlflow
8. create flask app which fetches promoted model from mlflow & make predictions on local system

![alt text](image.png)

9. Apply CD: dockerize app (for consistency so that code that works on my system will also work on cloud)
10. Push docker image to ECR and deploy on EC2/ECS
    + why deploy on ECS (advanced) then EC2?
    + EC2 is a single server single container deployment
    + It has scalability issues since its a single server single container
        - Vertical scaling: we can increase hardware of EC2
        - Horizontal scaling: we can increase servers (EC2 instance)count (Mostly preffered)
            - **PROBLEMS**
            - We'll have 2 run a lot of commands manually - chances of error increases
            - Traffic routing to different ips (since no load balancer setup right now)
            - We'll have to start & stop servers manually based on load - It should be done automatically
            - new model --> new docker image --> we'll have to update all servers manually
            - potential downtime
            - No health check whether server is working
            - No centralized logging & monitoring
            - Manual Security management 
            - Lack of validation check of health
            - CI/CD complexity manually - greate chances of error
    
    + **AMI (Amazon machine Images)** -  we can create a template which can be used to create multiple ec2 instances
        - select a running server
        - actions - image & templates - create image
        - image - AMIs - that AMI will be visible here
        - select & launch AMI instance
        - It will have all libraries & software pre-installed & docker image already available
        - Advance details - user data - add your script to run docker image
        - we can start new servers quickly with the help of AMI

    + **Traffic routing issue -- Load Balancer**
        - when we hit a website like facebook -- this rqst goes to DNS server & it will route it to thier ip address
        - problem in our case is since we have multiple servers running, which ip address should it send it to
        - for that we can use **load balancer** -- dns will send the rqst to load balancer which will decide where to send the rqst to
        - we can create load balancer & target group and attach all the servers
        - it will keep checking health of the servers & automatically route traffic
    
    + **Rigid setup -- ASG**
        - currently i have to check manually when to increase or decrease servers
        - we can use EC2 ASG (auto scaling group)
        - we can setup intial, min & max capacity, criteria to scale
        - it will check health & automatically remove-replace unhealthy server

    + **Docker image update**
        - we are changing previous model to a new one or change in UI

    + **Development strategy**

    + **Rollback**

    + **CI/CD**