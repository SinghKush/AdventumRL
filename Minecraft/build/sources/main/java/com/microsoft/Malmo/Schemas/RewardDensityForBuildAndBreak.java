//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2019.03.12 at 01:29:20 PM EDT 
//


package com.microsoft.Malmo.Schemas;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for RewardDensityForBuildAndBreak.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="RewardDensityForBuildAndBreak">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="PER_BLOCK"/>
 *     &lt;enumeration value="MISSION_END"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "RewardDensityForBuildAndBreak")
@XmlEnum
public enum RewardDensityForBuildAndBreak {

    PER_BLOCK,
    MISSION_END;

    public String value() {
        return name();
    }

    public static RewardDensityForBuildAndBreak fromValue(String v) {
        return valueOf(v);
    }

}
