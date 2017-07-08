# merger_detect_alg

  This is an algorithm to detect dark matter halo mergers from a halo catalog. Normally, dark matter mock simulations create halo catalogs that only record major mergers events (when a halo merges wih another halo whose mass is at least 30% of the first halo). When such an event occurs, there are very noticeable changes to various halo properties, leading additional research developments. However, such research efforts have been diluted with minor mergers (mergers with a halo less than 30% of the first halo's mass). It then becomes important to locate these minor merger events to have a more comprehensive view of dark matter halo property evolutions. Hence, this algorithm was created.
  
  Halo properties are very good indicators of merger events. While the algorithm allows the user to input a list of any property that he/she wants to use, the best best properties to look at are: mass, vmax, xoff, spin, and virial ratio. 
  
  In the end, the algorithm creates a new datafile of the original with additional columns added for the detected minor and major merger events.
