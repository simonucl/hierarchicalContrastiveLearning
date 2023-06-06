1.Title: BlurbGenreCollection-EN

2. Overview: 
The BlurbGenreCollection-EN (https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html) is a dataset consisting of advertising descriptions of books - so called blurbs for the English language. Each blurb is categorized into one or multiple categories. The categories are structured in a hierarchy. This dataset follows the policies as described in the RCV1 datset (http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf). We adapt RCV1's properties, which have been explained by its authors in detail and refer to their description. The minimum code policy requires the assignment of at least one category to each document of the collection. The hierarchy policy ensures that every ancestor of a document's label is assigned as well.

3. Copyright:
The copyright to all blurbs belongs to Penguin Random House (PRH), its licensors, vendors and/or its content providers since the blurbs were obtained through the penguinrandomhouse.com website. The blurbs serve promotional/public purposes and permission has been granted by Penguin Random House to share this dataset.
The dataset is shared under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/) which allows copying and redistributing the dataset as long as appropriate credit is given, especially to Penguin Random House and its content providers. In addition to indicating changes to the dataset, you may not use the dataset for commercial purposes.


4. Contents:
- /.BlurbGenreCollection_EN_train.txt: The training dataset contains 58,715 samples

- /.BlurbGenreCollection_EN.txt: The dev dataset, used as a validation dataset, contains 14,785 samples

- /.BlurbGenreCollection_EN.txt: The test dataset contains  18,394 samples

- /.hierarchy.txt: list of 'is a' pairwise relationships between genres, can be read as an acyclic graph


5. Data Format
Data format of the training, dev and test dataset is the same:
It uses xml tags as a structuring method.
< book> </book>: The scope of one blurb
<title> </title>: Title of the book the blurb is taken from
<body> </body>: Contains the actual blurb
<copyright> </copyright>: Statement regarding copyrights
<metadata> </metadata: All meta information are located within this scope
<topics> </topics>: Contains all extracted genres of a book
<d[n]> </d[n]>: Genre on the n-th depth of the genre hierarchy.
<author> </author>: Author of the book
<published> </published>: Date of publication
<isbn> </isbn>: Book Isbn
<url> </url>: Url of the page the blurb was taken from

4. Contact
If you are unsure about certain aspects of the data set, as well as methodology used, feel free to send an E-mail to Rami Aly:
5aly@informatik.uni-hamburg.de
