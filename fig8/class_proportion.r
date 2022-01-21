library(ggplot2)
d = read.csv("../fig7/protest_ARCH=vgg16_K=6_EPOCH=425_label_distance.txt", sep = "", header = F)
names(d) <- c("name", "class", "distance")


d = d[order(d$class, d$distance),]

library(dplyr)
d$class = as.character(d$class)
d$classname = recode(d$class, "0" = "1. Gathering at \ngovernment offices",
                     "1" = "2. Blocking streets", 
                     "2" = "3. Crowd gathering \nwith a zoom-in view",
                     "3" = "4. Screenshots of text",
                     "4" = "5. Crowd gathering\nwith police presence",
                     "5" = "6. Petition letters")

# dt = as.data.frame.table(table(d$class))
# names(dt) <- c("classno", "freq")
ggplot(d, aes(x = classname)) +
  geom_bar(aes(stat='count')) +
  geom_text(stat='count', aes(label=..count..), hjust=0.5)+
  coord_flip()+
  # coord_flip() + xlab("Number of clusters") + ylab("Percentage of images")+ 
  theme_classic(base_size=14)+
  theme(text=element_text(size=14),
        axis.ticks = element_blank(), axis.line = element_blank())
ggsave("class_proportion_protest.png")
