<?xml version="1.0" encoding="iso-8859-1"?> 
<xsl:stylesheet version="1.0" 
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"> 
<xsl:output method="html" encoding="utf-8" doctype-public="//W3C//DTD XHTML//EN" doctypesystem="http://www.w3.org/TR/2001/REC-xhtml11-20010531" indent="yes"/> 
<xsl:variable name="lowercase" select="'abcdefghijklmnopqrstuvwxyzéèêëïâôœæ'"/> 
 <xsl:variable name="uppercase" select="'ABCDEFGHIJKLMNOPQRSTUVWXYZÉÈÊËÏÂÔŒÆ'"/> 
 <!--fonction "romain" : convertit un chiffre arabe en chiffre 
romain pour afficher le numéro des actes--> 
 <xsl:template name="romain"> 
  <xsl:param name="arabe"/> 
  <xsl:choose> 
   <xsl:when test="$arabe=1">I</xsl:when> 
   <xsl:when test="$arabe=2">II</xsl:when> 
   <xsl:when test="$arabe=3">III</xsl:when> 
   <xsl:when test="$arabe=4">IV</xsl:when> 
<xsl:when test="$arabe=5">V</xsl:when>
   <xsl:otherwise> 
    <xsl:value-of select="$arabe"/> 
   </xsl:otherwise> 
  </xsl:choose> 
 </xsl:template> 
 <!--fonction "liaison" : pour une scène donnée, renvoie vrai ou 
faux selon que cette scène est liée ou non à la suivante--> 
 <xsl:template name="liaison"> 
  <xsl:param name="scène"/> 
  <xsl:if test="sp/@who = following
sibling::div2[1]/sp/@who"></xsl:if> 
 </xsl:template> 
 <xsl:template match="/"> 
  <html> 
   <head> 
    <meta http-equiv="Content-Type" content="text/html ; 
charset=utf-8"/> 
    <title> 
     <xsl:value-of 
select="TEI.2/teiHeader/fileDesc/titleStmt/title"/> 
    </title> 
    <link rel="stylesheet" type="text/css" 
href="style.css"/> 
   </head> 
   <body> 
    <table> 
     <!--ligne des actes--> 
     <tr> 
      <td rowspan="2" class="vide"/> 
      <xsl:for-each select="//div1"> 
       <xsl:variable name="n" select="count(div2)"/> 
       <xsl:choose> 
        <!--si on est dans le dernier acte--> 
        <xsl:when test="position() = last()"> 
         <td class="acte" colspan="{$n}"> 
          <xsl:call-template name="romain"> 
           <xsl:with-param name="arabe" 
select="@n"/> 
          </xsl:call-template> 
         </td> 
        </xsl:when> 
        <!--si l’on n’est dans le dernier acte --> 
        <xsl:otherwise> 
         <td class="dernier acte" colspan="{$n}"> 
          <!--<xsl:value-of select="@n"/>--> 
          <xsl:call-template name="romain"> 
           <xsl:with-param name="arabe" 
select="@n"/> 
          </xsl:call-template> 
         </td> 
        </xsl:otherwise> 
       </xsl:choose> 
      </xsl:for-each> 
     </tr> 
     <tr> 
      <xsl:for-each select="//div1"> 
       <xsl:choose> 
        <!--si l’on est dans le dernier acte--> 
        <xsl:when test="position() = last()"> 
         <xsl:for-each select="div2"> 
          <xsl:choose> 
           <!--si l’on n'est pas dans la dernière 
scène, appliquer la fonction "liaison"--> 
           <xsl:when test="position() != last()"> 
            <xsl:choose> 
             <!--si la scène est liée à la 
suivante : affichage normal--> 
             <xsl:when test="sp/@who = following-sibling::div2[1]/sp/@who"> 
              <td> 
               <xsl:value-of select="@n"/></td> 
             </xsl:when> 
             <xsl:otherwise> 
              <!-- si la scène n’est pas liée 
à la suivante : classe "rupture"--> 
              <td class="rupture"> 
               <xsl:value-of select="@n"/></td> 
             </xsl:otherwise> 
            </xsl:choose> 
           </xsl:when> 
           <xsl:otherwise> 
            <td> 
             <xsl:value-of select="@n"/></td> 
           </xsl:otherwise> 
          </xsl:choose> 
         </xsl:for-each> 
        </xsl:when> 
        <!--si l’on n’est dans le dernier acte --> 
        <xsl:otherwise> 
         <xsl:for-each select="div2"> 
          <xsl:choose> 
           <!--si c'est la dernière scène : class 
"rupture"--> 
           <xsl:when test="position() = last()"> 
            <td class="rupture"> 
             <xsl:value-of select="@n"/> 
            </td> 
           </xsl:when> 
           <!--si c'est une scène précédente : 
appliquer la fonction "liaison"--> 
           <xsl:otherwise> 
            <xsl:choose> 
             <!--si la scène est liée à la 
suivante : affichage normal--> 
             <xsl:when test="sp/@who = 
following-sibling::div2[1]/sp/@who"> 
              <td> 
               <xsl:value-of select="@n"/> 
              </td> 
             </xsl:when> 
             <xsl:otherwise> 
              <!-- si la scène n’est pas liée 
à la suivante : classe "rupture"--> 
              <td class="rupture"> 
               <xsl:value-of select="@n"/> 
              </td> 
             </xsl:otherwise> 
            </xsl:choose> 
           </xsl:otherwise> 
          </xsl:choose> 
         </xsl:for-each> 
        </xsl:otherwise> 
       </xsl:choose> 
      </xsl:for-each> 
     </tr> 
     <!--corps du tableau--> 
     <xsl:for-each select="//role"> 
      <xsl:variable name="role" select="@id"/> 
      <tr> 
       <td> 
        <xsl:value-of select="$role"/> 
       </td> 
       <!--afficher le statut du personnage pour chaque 
scène dans les colonnes suivantes --> 
       <xsl:for-each select="//div1"> 
        <xsl:choose> 
         <!--si on est dans le dernier acte--> 
         <xsl:when test="position() = last()"> 
          <xsl:for-each select="div2"> 
           <xsl:variable name="head" 
select="translate(head, $lowercase, $uppercase)"/> 
           <xsl:choose> 
            <!--si l’on n'est pas dans la 
dernière scène, appliquer la fonction "liaison"--> 
            <xsl:when test="position() != 
last()"> 
             <xsl:choose> 
              <!--si la scène est liée à la 
suivante : affichage normal--> 
              <xsl:when test="sp/@who = 
following-sibling::div2[1]/sp/@who"> 
               <xsl:choose> 
               <!--si le personnage ne parle 
pas : classe "muet" et valeur 0--> 
                <xsl:when 
test="contains($head, $role) and count(sp[@who=$role]) = 0"> 
                 <td class="muet">0</td> 
                </xsl:when> 
               <!--si le personnage parle : 
classe "parlant" et valeur 1--> 
                <xsl:when 
test="count(sp[@who=$role]) > 0"> 
                 <td 
class="parlant">1</td> 
                </xsl:when> 
               <!--si le personnage est 
absent : classe "absent" et pas de valeur--> 
                <xsl:otherwise> 
                 <td class="absent"/> 
                </xsl:otherwise> 
               </xsl:choose> 
              </xsl:when> 
              <xsl:otherwise> 
               <!--si la scène est liée à la 
suivante : classe "rupture" (puis déterminer le statut du 
personnage selon la même méthode)--> 
               <xsl:choose> 
                <xsl:when 
                 test="contains($head, 
$role) and count(sp[@who=$role]) = 0"> 
                 <td class="muet 
rupture">0</td> 
                </xsl:when> 
                <xsl:when 
test="count(sp[@who=$role]) > 0"> 
                 <td class="parlant 
rupture">1</td> 
                </xsl:when> 
                <xsl:otherwise> 
                 <td class="absent 
rupture"/> 
                </xsl:otherwise> 
               </xsl:choose> 
              </xsl:otherwise> 
             </xsl:choose> 
            </xsl:when> 
            <xsl:otherwise> 
             <xsl:choose> 
              <xsl:when test="contains($head, 
$role) and count(sp[@who=$role]) = 0"> 
               <td class="muet">0</td> 
              </xsl:when> 
              <xsl:when 
test="count(sp[@who=$role]) > 0"> 
               <td class="parlant">1</td> 
              </xsl:when> 
              <xsl:otherwise> 
               <td class="absent"/> 
              </xsl:otherwise> 
             </xsl:choose> 
            </xsl:otherwise> 
           </xsl:choose> 
           <xsl:variable name="head" 
select="translate(head, $lowercase, $uppercase)"/> 
           <xsl:choose> 
            <xsl:when test="contains($head, 
$role) and count(sp[@who=$role]) = 0"> 
             <td class="muet">0</td> 
            </xsl:when> 
            <xsl:when test="count(sp[@who=$role]) 
> 0"> 
             <td class="parlant">1</td> 
            </xsl:when> 
            <xsl:otherwise> 
             <td class="absent"/> 
            </xsl:otherwise> 
           </xsl:choose> 
          </xsl:for-each> 
         </xsl:when> 
         <xsl:otherwise> 
          <!--si l’on n’est pas dans le dernier 
acte : déterminer la liaison des scènes et le statut du personnage 
selon la même méthode--> 
          <xsl:for-each select="div2"> 
           <xsl:variable name="head" 
select="translate(head, $lowercase, $uppercase)"/> 
           <xsl:choose> 
            <!--si c'est la dernière scène : 
classe "rupture"--> 
            <xsl:when test="position() = last()"> 
             <xsl:choose> 
              <xsl:when test="contains($head, 
$role) and count(sp[@who=$role]) = 0"> 
               <td class="muet 
rupture">0</td> 
              </xsl:when> 
              <xsl:when test="count(sp[@who=$role]) > 0"> 
               <td class="parlant 
rupture">1</td> 
              </xsl:when> 
              <xsl:otherwise> 
               <td class="absent rupture"/> 
              </xsl:otherwise> 
             </xsl:choose> 
            </xsl:when> 
            <!--si c'est une scène précédente, 
appliquer la fonction "liaison"--> 
            <xsl:otherwise> 
             <xsl:choose> 
              <!--si la scène est liée à la 
suivante--> 
              <xsl:when test="sp/@who = following-sibling::div2[1]/sp/@who"> 
               <xsl:choose> 
                 
                <xsl:when test="contains($head, $role) and count(sp[@who=$role]) = 0"> 
                 <td class="muet">0</td> 
                </xsl:when> 
                <xsl:when test="count(sp[@who=$role]) > 0"> 
                 <td class="parlant">1</td> 
                </xsl:when> 
                <xsl:otherwise> 
                 <td class="absent"/> 
                </xsl:otherwise> 
               </xsl:choose> 
              </xsl:when> 
              <xsl:otherwise> 
               <!--si la scène n’est pas liée 
à la suivante : classe "rupture"--> 
               <xsl:choose> 
                <xsl:when 
                 test="contains($head, $role) and count(sp[@who=$role]) = 0"> 
                 <td class="muet rupture">0</td> 
                </xsl:when> 
                <xsl:when test="count(sp[@who=$role]) > 0"> 
                 <td class="parlant rupture">1</td> 
                </xsl:when> 
                <xsl:otherwise> 
                 <td class="absent rupture"/> 
                </xsl:otherwise> 
               </xsl:choose> 
              </xsl:otherwise> 
             </xsl:choose> 
             <xsl:choose> 
              <xsl:when test="contains($head, $role) and count(sp[@who=$role]) = 0"> 
               <td class="muet">a</td> 
              </xsl:when> 
              <xsl:when test="count(sp[@who=$role]) > 0"> 
               <td class="parlant">p</td> 
              </xsl:when> 
              <xsl:otherwise> 
               <td class="absent"/> 
              </xsl:otherwise> 
             </xsl:choose>
            </xsl:otherwise> 
           </xsl:choose> 
          </xsl:for-each> 
         </xsl:otherwise> 
        </xsl:choose> 
       </xsl:for-each> 
      </tr> 
     </xsl:for-each> 
    </table> 
   </body> 
  </html> 
 </xsl:template> 
</xsl:stylesheet>