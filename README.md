# Tensorflow JAVA Apache CXF example
A simple Apache CXF REST Server project loading and executing the SavedModel from the accompanying Jupyter example.

The pre-trained data from the Jupyter notebook is already included. The Server loads the SavedModel on startup. 
The REST call is tested with a simple integration test running over all lines of test data and comparing the REST
result with the result of the original python code. That way we can be sure that we did not forget any preprocessing, 
the export and integration went OK etc.

In a real world scenario, we could also add performance metrics here (e.g. F-Score). That way we can fully automate the
testing of the performance of our system and even automate the whole train-compile-test-deploy cycle. 

That way you can add training data in regular intervals to improve the system and automate deployment. If performance degrades 
to much, the Integration test will catch it and prevent deployment.

The code itself is under the Apache 2.0 license, the training data used for the model comes with the following note: 


> This dataset is public available for research. The details are described in *Cortez et al., 2009*. 
>  Please include this citation if you plan to use this database:
>
>  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
>  # Modeling wine preferences by data mining from physicochemical properties.
>  ## In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
> 
>  Available at: 
>   * [Elsevier](http://dx.doi.org/10.1016/j.dss.2009.05.016)
>   * [Pre-press (pdf)](http://www3.dsi.uminho.pt/pcortez/winequality09.pdf)
>   * [bib](http://www3.dsi.uminho.pt/pcortez/dss09.bib)

The trained data and the test set are added to the repo, you can override them by configuring the `beans.xml`
