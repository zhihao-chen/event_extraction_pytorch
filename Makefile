uninstall:
	pip uninstall unified_ie -y

install:
	make clean
	make uninstall
	pip install . --use-feature=in-tree-build
	make clean

clean:
	rm -rf build dist *.egg-info

build:
	python setup.py sdist bdist_wheel

upload:
	twine upload -r bailian dist/*

release:
	git tag `cat VERSION`
	git push origin `cat VERSION`

release-again:
	git tag -d `cat VERSION`
	git push -d origin `cat VERSION`
	git tag `cat VERSION`
	git push origin `cat VERSION`
