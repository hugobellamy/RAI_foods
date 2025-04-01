An LLM system to generate accurate diet information from simple meal descriptions. 

Provides both macro and micro nutrient content of food. Uses a 2 stage approach:
1 - break down meal into simple components and their weights
2 - look these up in a database and select the most appropriate 

The LLM performs both these steps and by seperating the process into two steps much more accurate diet information is returned. 

This system could easily be extended to use pictures by using an LLM to describe an image of food and estimate the weight of each component of the meal. 

Current version uses OpenAI models but could easily be adapted to use others. 

Diet data is from USDA http://www.ars.usda.gov/nea/bhnrc/fsrg. Many other sources do not provide micro nutrient details.

