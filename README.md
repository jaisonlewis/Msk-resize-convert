# Msk-resize-convert
Resizes images, converts to png, attaches masks using coordinates from CSV file, creates labels text files from CSV data.

Takes Images from a directory, resizes it from 512X512 to 256X256. 
Then extracts coordinates from CSV file, adapts it to new size and uses it to create a mask file
Saves all image files as PNG
Saves labels for coordinates into a text file. 
