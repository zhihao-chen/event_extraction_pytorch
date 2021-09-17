# -*- coding: utf8 -*-
"""
======================================
    Project Name: EventExtraction
    File Name: setup.py
    Author: czh
    Create Date: 2021/9/16
--------------------------------------
    Change Activity: 
======================================
"""
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get version from the VERSION file
with open(path.join(here, "VERSION"), encoding="utf-8") as f:
    version = f.read().strip()

# 改成自己的工程名，尽量与gitlab的工程名起名一致
proj_name = "EventExtraction"
keywords = "event_extraction, bert_crf"
git_url = f"https://gitlab.bailian-ai.com/ai_algo/{proj_name}"


def load_requirements(file_name="requirements.txt", comment_char="#"):
    with open(path.join(here, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(
    name=proj_name,
    author="bailian.ai",
    author_email="chenzhihao@bailian.ai",
    description=long_description,
    version=version,
    url=git_url,
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",

        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",

        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by "pip install". See instead
        # "python_requires" below.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],

    keywords=keywords,
    packages=find_packages(
        exclude=[
            "notebooks",
            "data",
            "conf",
            "examples",
            "docs",
            "tests",
            "contrib",
            "scripts",
        ]
    ),
    python_requires=">=3.6, <4",
    install_requires=load_requirements(),
    extras_require={  # Optional
        "dev": [],
        "test": ["pytest"],
    },

    include_package_data=True,
    project_urls={  # Optional
        "Source": git_url,
    },

)
