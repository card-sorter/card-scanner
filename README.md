# card-scanner

## Information
Scanning cards is a problem that has been [solved before](https://github.com/jslok/card-scanner), but it has never been very accessible to the public to use and implement in their projects for free. This aims to solve that problem.

## How it works
We will use image hashing, as it is a generally well defined method of solving this exact problem. It's how the [Physbatch 9000 scans cards](https://tcgmachines.com/blog/our-robots-can-read), and it's how jslok above solved the problem above. We hash the cards from the card database, and we hash the card that has been cropped and perspective transformed, and we compare all of the hashes to each other. We can do this through a SQL query so we don't need to worry much about storing the data manually, and it will also efficiently use the resources that are alloted to it. This way we don't need to optimize the slow parts of the program.

We start by building our image database. We first fetch all of the card information from [TCGCSV](https://www.tcgcsv.com) as it's an easy source of the card images, alongside the card data. We store this data into a MySQL database and also download the images so that we can hash them. Once we hash the images, we put those hashes into a separate table using the product ID as the primary key and having several other bigint columns where we store parts of the hash in. This is because running the query to calculate the Hamming distance is far [more efficient if the data is in a bigint compared to in a string binary for some reason](https://stackoverflow.com/a/4783415). 

To scan a picture of a card, we first have to crop and perspective transform the card so that it can match with an image in our database. We can do this using a YOLOv11 model trained to mask out cards(see jslok's repo for more information). Once it's been cropped, it just needs to be hashed and compared in the MySQL database and the product ID that closest matches the scan should be returned. We can then fetch and return the card data from the database.

Some issues with this is that it might have difficulty discerning between different printings of a card ([Ultra Ball FLF](https://www.tcgplayer.com/product/91236) and [Ultra Ball FCO](https://www.tcgplayer.com/product/117886) for example). We can cross this road once we get there, but it is likely that we will need to implement OCR of some sorts when this case occurrs.
