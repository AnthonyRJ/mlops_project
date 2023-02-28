pipeline {
    agent any
    environment {
        PATH = "C:/WINDOWS/SYSTEM32;C:/Users/Armand/AppData/Local/Programs/Python/Python38;C:/Program Files/Docker/Docker/resources/bin"
        DOCKER_USERNAME = "necroz"
        DOCKER_PASSWORD = ""
    }
    stages {
        stage('Checkout'){
            steps{
                echo 'Creating final to branch to merge with main...'
                sh """ 
                  git checkout -b finalBranch
                  git push -u origin finalBranch
                """
            }
        }
        stage('Test') {
            steps {
                echo 'Testing...'
                sh """
                    bat 'python -m pip install --upgrade pip'
                    bat 'python -m pip install Flask'
                    bat 'python -m pip install requests'
                    bat 'python test_main.py'
                    bat 'python test_endtoend.py'
                """
            }
        }
        stage('Build docker image') {
            steps {
                echo 'Building Dodcker image...'
                
                sh """
                    docker build -t mlops-flask-app .
                    docker run -d mlops-flask-app
                    docker ps
                """

            }
        }
        stage("Docker Push") {
            steps {
                sh """
                    bat 'docker login --username=$DOCKER_USERNAME --password=$DOCKER_PASSWORD'
                    bat 'docker tag mlops-flask-app necroz/mlopsrepo'
                    bat 'docker push necroz/mlopsrepo'
                """
                
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
