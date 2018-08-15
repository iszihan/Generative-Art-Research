void setup(){
  size(224,224);
  noLoop();
  
  
}

void draw(){
  for (int itr = 0; itr<860; itr++){
    clear();
    float back_col = random(0,255);
    background(back_col);
    PImage temp = get();
    String filename = String.format("Output/tri-r%02d-bl%03d.png", 00, itr);
    temp.save(filename);

  
  }


}
