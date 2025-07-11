setwd("C:/Graham/cardiotoxicity/Part3 Herbicide")

transpose1<- read.csv("Herbicide3 vs VEH Cleaned.csv", colClasses = "character")

sucess1<-t(transpose1)

write.table(sucess1, file= "Herbicide3 vs VEH Cleaned.txt", quote= FALSE, col.names=FALSE, sep = "," )
