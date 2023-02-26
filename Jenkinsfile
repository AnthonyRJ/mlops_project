pipeline {
    agent any
    environment {
        PATH = "C:/WINDOWS/SYSTEM32;C:/Users/Armand/AppData/Local/Programs/Python/Python38;C:/Program Files/Docker/Docker/resources/bin"
        DOCKER_IMAGE = "docker-image-name"
        DOCKER_USERNAME = ""
        DOCKER_PASSWORD = ""
        DOCKER_REPO = ""
        DOCKER_TAG = ""
    }
    stages {
        stage('Checkout'){
            steps{
                echo 'Creating final to branch to merge with main...'
                sh """ 
                  # Complete here
                """
            }
        }
        stage('Build') {
            steps {
                echo 'Building...'
                sh """
                    # Complete here
                """
            }
        }
        stage('Test') {
            steps {
                echo 'Testing...'
                sh """
                    # Complete here
                """
            }
        }
        stage('Build docker image') {
            steps {
                echo 'Building Dodcker image...'
                
                sh """
                    docker build -t $DOCKER_IMAGE
                    docker ps
                """

            }
        }
        stage("Docker Push") {
            steps {
                bat 'docker login --username=$DOCKER_USERNAME --password=$DOCKER_PASSWORD'
                bat 'docker tag jenkins-flask-app $DOCKER_REPO'
                bat 'docker push $DOCKER_TAG'
                
            }
        }
        stage('User Acceptance on Release') {
            steps {
                echo 'Waiting for User Acceptance...'
                input {
                    message "Do you accept this release for deployment?"
                }
            }
        }
      stage('Merge branches'){
          steps{
              echo 'Merge final branch with main...'
              sh """
                git checkout main
                git pull origin main
                git merge finalBranch
                git push -u origin main
              """
          }
      }
    }
}
