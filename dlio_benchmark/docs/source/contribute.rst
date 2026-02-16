Contributing Guide
========================

Testing
------------------------
All help is appreciated! If you're in a position to run the latest code, consider helping us by reporting any functional problems, performance regressions, or other suspected issues. By running the latest code on a wide range of realistic workloads, configurations, and architectures we're better able to quickly identify and resolve issues.

Reporting Bugs
-----------------
You can submit bug report in the `issue tracker`_.  Please search the `issue tracker`_ first to ensure the issue hasn't been reported before. Open a new issue only if you haven't found anything similar to your issue.

.. note::

    When opening a new issue, please include the following information at the top of the issue:

    * What operating system (with version) you are using
    * The DLIO version you are using
    * Describe the issue you are experiencing
    * Describe how to reproduce the issue
    * Include any warnings or errors
    * Apply any appropriate labels, if necessary

Developing New Features
------------------------
We welcome the contribution from the community for developing new features of the benchmark. Specifically, we welcome contribution in the following aspects: 

* Support for new workloads: if you think that your workload(s) would be interested to the public, and would like to provide the yaml file to be included in the repo, please submit an issue in the `issue tracker`_. Please also include the link to the real workload github repo. 
* Support for loading new data formats.
* Support for new data loaders, such as DALI loader, MxNet loader, etc
* Support for new frameworks, such as MxNet. 
* Support for noval file or storage systems, such as AWS S3.

If there are other features that you think would be great to have in DLIO, please submit an issue with label ``feature request``. 

For developing all these features, if you think that it will have significant impact on the original structure of the code, please submit an issue to the `issue tracker`_ first, and contact ALCF DLIO `mailing list`_ to discuss before proceeding further. This is to minize the effort involved in merging the pull request. 

Pull Requests
------------------------
* In the pull request, please include a comment in the pull request, mentioning the following information 
    - what new feature(s) has been added or what problem has been solved. 
    - what are the major changes to the code. 
    - what potential issues or limitations it will cause if there is any
* All pull requests must be based on the current main branch and apply without conflicts.
* Try to keep pull requests simple. Simple code with comments is much easier to review and approve.
* Test cases should be provided when appropriate.
* If your pull request improves performance, please include some benchmark results.
* The pull request must pass all regression tests before being accepted.
* All proposed changes must be approved by a DLIO project member.

.. explicit external hyperlink targets

.. _mailing list: huihuo.zheng@anl.gov
.. _issue tracker: https://github.com/argonne-lcf/dlio_benchmark/issues