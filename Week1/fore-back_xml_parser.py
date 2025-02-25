from bs4 import BeautifulSoup
from lxml import etree
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_xml', required=False, default="../ai_challenge_s03_c010-full_annotation.xml", help='Input xml')
    parser.add_argument('--new_xml', required=False, default="./week1_anot.xml", help='Input xml')


    result_struct = {}
    # Parse the arguments and call the appropriate function
    args = parser.parse_args()
    with open(args.path_xml, "r", encoding="utf-8") as file:
        xml_content = file.read()
    
    soup = BeautifulSoup(xml_content, "xml")
    boxes = soup.find_all("box")
    width = (soup.find("width").text)
    height = (soup.find("height").text)
    
    for box in boxes:
        parked_tag = box.find("attribute", {"name": "parked"})
        is_parked = parked_tag and parked_tag.text.strip().lower() == "true"

        if not is_parked: # so its either a bike or not a car
            frame = box.get("frame")
            if not frame in result_struct:
                result_struct[frame] = []
            result_struct[frame].append(box)
    
    # Now create a new XML structure with frames as the first level
    root = etree.Element("annotations")
    etree.SubElement(root, "meta", width=width, height=height)

    for frame, boxes in result_struct.items():
        frame_element = etree.SubElement(root, "frame", number=frame)

        for box in boxes:
            # Create a new <box> element and copy its attributes
            box_element = etree.SubElement(frame_element, "box", frame=box.get("frame"), 
                                          xtl=box.get("xtl"), ytl=box.get("ytl"),
                                          xbr=box.get("xbr"), ybr=box.get("ybr"),
                                          outside=box.get("outside"), occluded=box.get("occluded"),
                                          keyframe=box.get("keyframe"))

    # Save the new XML file
    tree = etree.ElementTree(root)
    tree.write(args.new_xml, pretty_print=True, xml_declaration=True, encoding="UTF-8")

    print("Filtered XML created successfully!")
