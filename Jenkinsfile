pipeline {
    agent { node { label 'Linux_Default' } }
    stages {
        stage('Build') {
            steps {
                echo 'installing package with poetry'
		        sh './install.sh'
            }
        }
        stage('Test') {
            steps {
                echo 'testing'
		        sh './test_package.sh'
            }
        }
    }
}
