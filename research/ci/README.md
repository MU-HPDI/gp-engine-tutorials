# CI/CD

This directory contains samples and slides for using GitLab CI to build and publish docker containers for use on Nautilus.

For each example directory:

1. Create a new repository on the [NRP GitLab](gitlab.nrp-nautilus.io)
1. Clone the empty directory: `git clone https://gitlab.nrp-nautilus.io/[username]/[repository].git`
1. Copy the files from the sample directory to your repository directory: `cp ./python-ci /path/to/my/repo`
1. Rename the GitLab YAML from `gitlab-ci.yml` to `.gitlab-ci.yml` in the base directory of the repository: `mv /path/to/my/repo/gitlab-ci.yml /path/to/my/repo/.gitlab-ci.yml`
1. Commit and Push the files to NRP GitLab
1. Use the published container in your pods/jobs. Each example directory has a `sample_pod.yml` that you can refer to as well
