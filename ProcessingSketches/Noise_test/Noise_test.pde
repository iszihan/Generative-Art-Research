PImage img;
PGraphics mask_hori, mask_vert;

void setup(){
  size(224, 224); 
  img = loadImage("Noise/img-f00-000.png");
  mask_hori = createGraphics(224,224);
  mask_hori.beginDraw();
  mask_hori.rect(0, 0, img.width/2, img.height);
  mask_hori.endDraw(); 
  
  mask_vert = createGraphics(224,224);
  mask_vert.beginDraw();
  mask_vert.rect(0, 0, img.width, img.height/2);
  mask_vert.endDraw(); 
  noLoop();
}
 
 
void draw(){ 
  
  imageFlipped_hori(img, 0, 0 );  
  
  img.mask(mask_hori);
  image(img,0,0);
  String fileName = String.format("Sym/mask_hori.png");
  save(fileName);
  
  img = loadImage("Noise/img-f00-000.png");
  imageFlipped_vert(img, 0, 0 );  
  img.mask(mask_vert);
  image(img,0,0);
  fileName = String.format("Sym/mask_vert.png");
  save(fileName);
  
  
}

void imageFlipped_vert( PImage img, float x, float y ){
  pushMatrix(); 
  translate(0, img.height);
  scale( 1, -1 );
  image(img, x, y); 
  popMatrix(); 
} 

void imageFlipped_hori( PImage img, float x, float y ){
  pushMatrix(); 
  translate(img.width, 0);
  scale( -1, 1 );
  image(img, x, y); 
  popMatrix(); 
} 
