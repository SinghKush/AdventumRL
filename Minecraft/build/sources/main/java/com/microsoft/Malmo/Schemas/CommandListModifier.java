//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2019.03.12 at 01:29:20 PM EDT 
//


package com.microsoft.Malmo.Schemas;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;


/**
 * 
 *         Allow-list/deny-list base class - restricted by each command handler to only allow that handler's commands.
 *       
 * 
 * <p>Java class for CommandListModifier complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CommandListModifier">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;choice maxOccurs="unbounded">
 *         &lt;element name="command" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/choice>
 *       &lt;attribute name="type" default="deny-list">
 *         &lt;simpleType>
 *           &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *             &lt;enumeration value="deny-list"/>
 *             &lt;enumeration value="allow-list"/>
 *           &lt;/restriction>
 *         &lt;/simpleType>
 *       &lt;/attribute>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CommandListModifier", propOrder = {
    "command"
})
@XmlSeeAlso({
    com.microsoft.Malmo.Schemas.ContinuousMovementCommands.ModifierList.class,
    com.microsoft.Malmo.Schemas.ChatCommands.ModifierList.class,
    com.microsoft.Malmo.Schemas.SimpleCraftCommands.ModifierList.class,
    com.microsoft.Malmo.Schemas.AbsoluteMovementCommands.ModifierList.class,
    com.microsoft.Malmo.Schemas.DiscreteMovementCommands.ModifierList.class,
    com.microsoft.Malmo.Schemas.InventoryCommands.ModifierList.class,
    com.microsoft.Malmo.Schemas.MissionQuitCommands.ModifierList.class,
    com.microsoft.Malmo.Schemas.HumanLevelCommands.ModifierList.class
})
public class CommandListModifier {

    protected List<String> command;
    @XmlAttribute(name = "type")
    protected String type;

    /**
     * Gets the value of the command property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the command property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getCommand().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getCommand() {
        if (command == null) {
            command = new ArrayList<String>();
        }
        return this.command;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getType() {
        if (type == null) {
            return "deny-list";
        } else {
            return type;
        }
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setType(String value) {
        this.type = value;
    }

}
