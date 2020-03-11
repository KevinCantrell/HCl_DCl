from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy import stats
#import Tkinter as tk
#import tkFileDialog
try:
    import Tkinter as tk
    from tkFileDialog import askopenfilename
    from tkFileDialog import asksaveasfilename
except ImportError:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    from tkinter.filedialog import asksaveasfilename
from scipy.signal import savgol_filter
from matplotlib.widgets import Button

#Things to do:

#write to the excel file a nicely formatted and labeled table with the following constants
mH=1.07825 #g/mol
mD=2.014102
mCl35=34.968853 #75.78%
mCl37=36.965903 #24.22%
Na=6.022140857E23  #(mol-1)
h=6.62607015E-27 #(cm^2*g/s, 6.62607015E-34 Js *1000 g/kg * 100 cm/m * 100 cm/m)
c=2.99792458E10 #(cm/s)

#add formulas to the Excel file to calculate the masses of the individual isotopes and the reduced masses of the molecules in units of grams/atom and grams/molecule respectively

#add formulas to calculate D, a, B, and vo based on the fit coefficients. Make sure these calculations have clear labels and units (wavenumbers for all four of these)
#wavenumber(as a function of index, m) = (-4D)m^3 + (-a)m^2 + (2B-2a)m^1 + (vo)m^0

#add formulas to calculate I (the moment of intertia in g*cm^2) and r (the internuclear separation in angstroms).
#I=h/(8*pi^2*c*B) (g*cm^2). I=u*r^2

#output FitTable so all the coefficients are in the first four rows and the corresponding errors are in the next four rows

#add an additional column to the PeakTable that is the experimentally measured wavenumber of the peak (not the interpolated value based on the derivative)

#add annotation to the plot that labels each peak with its corresponding index
#make a button that toggles these labels on and off 

#add interaction with user (in the beginning) to differentiate HCl from DCl, P from R, and Cl35 from Cl37    

def PolyReg(X,Y,order):
    """
    Perform a least squares polynomial fit
    
    Parameters
    ----------
        X: a numpy array with shape M
            the independent variable 
        Y: a numpy array with shape M
            the dependent variable
        order: integer
            the degree of the fitting polynomial
    
    Returns
    -------
    a dict with the following keys:
        'coefs': a numpy array with length order+1 
            the coefficients of the fitting polynomial, higest order term first
        'errors': a numpy array with length order+1
            the standard errors of the calculated coefficients, 
            only returned if (M-order)>2
        'sy': float
            the standard error of the fit
        'n': integer
            number of data points (M)
        'poly':  class in numpy.lib.polynomial module
            a polynomial with coefficients (coefs) and degreee (order),
            see example below
        'res': a numpy array with length M
            the residuals of the fit
    
    Examples
    --------
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> fit = PolyReg(x, y, 2)
    >>> fit
    {'coefs': array([-0.16071429,  0.50071429,  0.22142857]),
     'errors': array([0.06882765, 0.35852091, 0.38115025]),
     'n': 6,
     'poly': poly1d([-0.16071429,  0.50071429,  0.22142857]),
     'res': array([-0.22142857,  0.23857143,  0.32      , -0.17714286, -0.45285714,
         0.29285714]),
     'sy': 0.4205438655564278}
    
    It is convenient to use the "poly" key for dealing with fit polynomials:
    
    >>> fit['poly'](0.5)
    0.43160714285714374
    >>> fit['poly'](10)
    -10.842857142857126
    >>> fit['poly'](np.linspace(0,10,11))
    array([  0.22142857,   0.56142857,   0.58      ,   0.27714286,
        -0.34714286,  -1.29285714,  -2.56      ,  -4.14857143,
        -6.05857143,  -8.29      , -10.84285714])
    """
    n=len(X)
    if X.shape!=Y.shape:
        raise Exception('The shape of X and Y should be the same')
    df=n-(order+1)
    if df<0:
        raise Exception('The number of data points is too small for that many coefficients')
    #if df = 0, 1, or 2 we call numpy's polyfit function without calculating the covariance matrix
    elif df<(3):
        coefs=np.polyfit(X,Y,order)
        p=np.poly1d(coefs)
        yFit=p(X)
        res=Y-yFit
        sy=np.sqrt( np.sum(res**2) / df )
        if order==1:
            #if the fit is linear we can explicitly calculate the standard errors of the slope and intercept
            #http://www.chem.utoronto.ca/coursenotes/analsci/stats/ErrRegr.html
            stdErrors=np.zeros((2))
            xVar=np.sum((X-np.mean(X))**2)
            sm=sy/np.sqrt(xVar)
            sb=np.sqrt(np.sum(X**2)/(n*xVar))*sy
            stdErrors[0]=sm
            stdErrors[1]=sb            
        else:
            stdErrors=np.full((order+1),np.inf)
    else:
        #The diagonal of the covariance matrix is the square of the standard error for each coefficent
        #NOTE 1: The polyfit function conservatively scales the covariance matrix. Dividing by (n-# coefs-2) rather than (n-# coefs)
        #NOTE 2: Because of this scaling factor, you can get division by zero in the covariance matrix when (# coefs-n)<2
        coefs,cov=np.polyfit(X,Y,order,cov=True)
        p=np.poly1d(coefs)
        yFit=p(X)
        res=Y-yFit
        sy=np.sqrt( np.sum(res**2) / df )
        stdErrors=np.sqrt(np.diagonal(cov)*(df-2)/df)
    return {'coefs':coefs,'errors':stdErrors,'sy':sy,'n':n,'poly':p,'res':res}

def FormatSciUsingError(x,e,withError=False,extraDigit=0):
    """
    Format the value, x, as a string using scientific notation and rounding appropriately based on the absolute error, e
    
    Parameters
    ----------
        x: number
            the value to be formatted 
        e: number
            the absolute error of the value
        withError: bool, optional
            When False (the default) returns a string with only the value. When True returns a string containing the value and the error
        extraDigit: int, optional
            number of extra digits to return in both value and error
    
    Returns
    -------
    a string
    
    Examples
    --------
    >>> FormatSciUsingError(3.141592653589793,0.02718281828459045)
    '3.14E+00'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045)
    '3.142E+00'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045,withError=True)
    '3.142E+00 (+/- 3E-03)'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045,withError=True,extraDigit=1)
    '3.1416E+00 (+/- 2.7E-03)'
    >>> FormatSciUsingError(123456,123,withError=True)
    '1.235E+05 (+/- 1E+02)'
    """
    if abs(x)>=e:
        NonZeroErrorX=np.floor(np.log10(abs(e)))
        NonZeroX=np.floor(np.log10(abs(x)))
        formatCodeX="{0:."+str(int(NonZeroX-NonZeroErrorX+extraDigit))+"E}"
        formatCodeE="{0:."+str(extraDigit)+"E}"
    else:
        formatCodeX="{0:."+str(extraDigit)+"E}"
        formatCodeE="{0:."+str(extraDigit)+"E}"
    if withError==True:
        return formatCodeX.format(x)+" (+/- "+formatCodeE.format(e)+")"
    else:
        return formatCodeX.format(x)

def AnnotateFit(fit,axisHandle,annotationText='Eq',color='black',arrow=False,xArrow=0,yArrow=0,xText=0.5,yText=0.2,boxColor='0.9'):
    """
    Annotate a figure with information about a PolyReg() fit
    
    see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html
    https://matplotlib.org/examples/pylab_examples/annotation_demo3.html
    
    Parameters
    ----------
        fit: dict, returned by the function PolyReg(X,Y,order)
            the fit to be summarized in the figure annotation 
        axisHandle: a matplotlib axes class
            the axis handle to the figure to be annotated
        annotationText: string, optional
            When "Eq" (the default) displays a formatted polynomial with the coefficients (rounded according to their error) in the fit. When "Box" displays a formatted box with the coefficients and their error terms.  When any other string displays a text box with that string.
        color: a valid color specification in matplotlib, optional
            The color of the box outline and connecting arrow.  Default is black. See https://matplotlib.org/users/colors.html
        arrow: bool, optional
            If True (default=False) draws a connecting arrow from the annotation to a point on the graph.
        xArrow: float, optional 
            The X coordinate of the arrow head using units of the figure's X-axis data. If unspecified or 0 (and arrow=True), defaults to the center of the X-axis.
        yArrow: float, optional 
            The Y coordinate of the arrow head using units of the figure's Y-axis data. If unspecified or 0 (and arrow=True), defaults to the calculated Y-value at the center of the X-axis.
        xText: float, optional 
            The X coordinate of the annotation text using the fraction of the X-axis (0=left,1=right). If unspecified, defults to the center of the X-axis.
        yText: float, optional 
            The Y coordinate of the annotation text using the fraction of the Y-axis (0=bottom,1=top). If unspecified, defults to 20% above the bottom.
    
    Returns
    -------
    a dragable matplotlib Annotation class
    
    Examples
    --------
    >>> annLinear=AnnotateFit(fitLinear,ax)
    >>> annLinear.remove()
    """
    c=fit['coefs']
    e=fit['errors']
    t=len(c)
    if annotationText=='Eq':
        annotationText="y = "
        for order in range(t):
            exponent=t-order-1
            if exponent>=2:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])+"x$^{}$".format(exponent)+" + "
            elif exponent==1:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])+"x + "
            else:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])
        annotationText=annotationText+", sy={0:.1E}".format(fit['sy'])
    elif annotationText=='Box':
        annotationText="Fit Details:\n"
        for order in range(t):
            exponent=t-order-1
            annotationText=annotationText+"C$_{x^{"+str(exponent)+"}}$ = "+FormatSciUsingError(c[order],e[order],extraDigit=1)+' $\pm$ '+"{0:.1E}".format(e[order])+'\n'
        annotationText=annotationText+'n = {0:d}'.format(fit['n'])+', DoF = {0:d}'.format(fit['n']-t)+", s$_y$ = {0:.1E}".format(fit['sy'])
    if (arrow==True):
        if (xArrow==0):
            xSpan=axisHandle.get_xlim()
            xArrow=np.mean(xSpan)
        if (yArrow==0):    
            yArrow=fit['poly'](xArrow)
        annotationObject=axisHandle.annotate(annotationText, 
                xy=(xArrow, yArrow), xycoords='data',
                xytext=(xText, yText),  textcoords='axes fraction',
                arrowprops={'color': color, 'width':1, 'headwidth':5},
                bbox={'boxstyle':'round', 'edgecolor':color,'facecolor':boxColor}
                )
    else:
        xSpan=axisHandle.get_xlim()
        xArrow=np.mean(xSpan)
        ySpan=axisHandle.get_ylim()
        yArrow=np.mean(ySpan)
        annotationObject=axisHandle.annotate(annotationText, 
                xy=(xArrow, yArrow), xycoords='data',
                xytext=(xText, yText),  textcoords='axes fraction',
                ha="left", va="center",
                bbox={'boxstyle':'round', 'edgecolor':color,'facecolor':boxColor}
                )
    annotationObject.draggable()
    return annotationObject

def Cross(x, y, crossPoint=0, direction='cross'):
    """
    Given a Series returns all the index values where the data values equal 
    the 'cross' value. 

    Direction can be 'rising' (for rising edge), 'falling' (for only falling 
    edge), or 'cross' for both edges
    """
    # Find if values are above or bellow yvalue crossing:
    above=y > crossPoint
    below=np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]

    # Calculate x crossings with interpolation using formula for a line:
    x1 = x[idxs]
    x2 = x[idxs+1]
    y1 = y[idxs]
    y2 = y[idxs+1]
    x_crossings = (crossPoint-y1)*(x2-x1)/(y2-y1) + x1

    return x_crossings,idxs

def PeakFind(SegmentX,SegmentY,peakHeightThreshold=0):
    """
    Finds peaks in a data array based on zero crossing in the derivative (calculated with numpy's gradient function)
    
    see https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html
           
    Parameters
    ----------
        SegmentX: a numpy array
            The x-cordinates of the data in Segment Y,  does not have to be evenly spaced
        SegmentY: a numpy array
            The data to find peaks in
        peakHeightThreshold: number, optional
            Only peaks with whre the y-values minus the averge of two adjacent valleys is greater than the threshold will be returned
    
    Returns
    -------
        peakX: a numpy array
            the interpolated x-coordinate of the peaks in y
        peakY: a numpy array
            the y-coordinate of the peaks
        peakHeight: a numpy array
            the heights of the peaks relative to the average y-coordinate of the two adjacent valleys 
    
    Examples
    --------
    >>> SegmentY=np.array([1,0,3,2,1,4,1,0,1])
    >>> SegmentX=np.array([1,2,3,4,5,6,7,8,9])
    >>> PeakFind(SegmentX,SegmentY)
    (array([3.5, 6. ]), array([3, 4]), array([2.5, 3.5]))
    >>> peakX,peakY,peakHeight=PeakFind(SegmentX,SegmentY)

    >>> plt.plot(SegmentX,SegmentY,'-ok')
    >>> plt.plot(peakX,peakY,'^r')
    """
    
    deriv1=np.gradient(SegmentY, SegmentX, edge_order=1)
    #deriv1sm=savgol_filter(SegmentY, 11, 2,deriv=1)
    valleyX,vindex = Cross(SegmentX,deriv1,direction='rising')
    peakX,pindex = Cross(SegmentX,deriv1,direction='falling')
    valleyY=np.min([SegmentY[vindex],SegmentY[vindex+1]],axis=0)
    peakY=np.max([SegmentY[pindex],SegmentY[pindex+1]],axis=0)
    midPeaksBool=((peakX>=valleyX[0]) & (peakX<=valleyX[-1]))
    peakX=peakX[midPeaksBool]
    peakY=peakY[midPeaksBool]
    valleyMean=valleyY[0:-1]+(np.diff(valleyY)/2)
    peakHeightBool=(peakY-valleyMean)>=peakHeightThreshold
    peakX=peakX[peakHeightBool]
    peakY=peakY[peakHeightBool]
    valleyMean=valleyMean[peakHeightBool]
    return peakX,peakY,peakY-valleyMean

def reducedMass(m1,m2):
    if m1>0.1:
        #to grams per atom
        m1=m1/Na
        m2=m2/Na
        #to kg per atom
        #m1=m1/1000
        #m2=m2/1000
    return (m1*m2)/(m1+m2)

def waveNumber(k,u):
    freq=1/(2*np.pi)*np.sqrt(k/u)
    wavenumber=freq/c
    return wavenumber

def updateSpectra():
    global fig,ax,h_cl37,h_cl35,d_cl35,d_cl37,h_low,d_low
    h_cl37.remove()
    h_cl35.remove()
    d_cl35.remove()
    d_cl37.remove()
    h_low.remove()
    d_low.remove()
    h_cl35,=ax.plot(PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35')]['Wavenumber'],PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35')]['Absorbance'],'o', color='tab:red',picker=10)
    h_cl37,=ax.plot(PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37')]['Wavenumber'],PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37')]['Absorbance'],'o',color='tab:green', picker=10)
    d_cl35,=ax.plot(PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35')]['Wavenumber'],PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35')]['Absorbance'],'^', color='tab:red', picker=10)
    d_cl37,=ax.plot(PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37')]['Wavenumber'],PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37')]['Absorbance'],'^',color='tab:green', picker=10)
    d_low,=ax.plot(PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='low')]['Wavenumber'],PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='low')]['Absorbance'],'^',color='lightgrey', picker=10)
    h_low,=ax.plot(PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='low')]['Wavenumber'],PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='low')]['Absorbance'],'o',color='lightgrey', picker=10)
    fig.canvas.draw()
    fig.canvas.flush_events()

def indexPeaks():
    global PeakTable
    PeakTable=PeakTable.sort_values(by=['Wavenumber'])
    PeakTable['Counter']=0
    Rsize=PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35') & (PeakTable['Rot']=='R')].Wavenumber.count()
    Psize=PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35') & (PeakTable['Rot']=='P')].Wavenumber.count()
    PeakTable.loc[((PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35') & (PeakTable['Rot']=='P')),'Counter']=np.arange(-Psize,0)
    PeakTable.loc[((PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35') & (PeakTable['Rot']=='R')),'Counter']=np.arange(1,Rsize+1)
    Rsize=PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37') & (PeakTable['Rot']=='R')].Wavenumber.count()
    Psize=PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37') & (PeakTable['Rot']=='P')].Wavenumber.count()
    PeakTable.loc[((PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37') & (PeakTable['Rot']=='P')),'Counter']=np.arange(-Psize,0)
    PeakTable.loc[((PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37') & (PeakTable['Rot']=='R')),'Counter']=np.arange(1,Rsize+1)
    Rsize=PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35') & (PeakTable['Rot']=='R')].Wavenumber.count()
    Psize=PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35') & (PeakTable['Rot']=='P')].Wavenumber.count()
    PeakTable.loc[((PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35') & (PeakTable['Rot']=='P')),'Counter']=np.arange(-Psize,0)
    PeakTable.loc[((PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35') & (PeakTable['Rot']=='R')),'Counter']=np.arange(1,Rsize+1)
    Rsize=PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37') & (PeakTable['Rot']=='R')].Wavenumber.count()
    Psize=PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37') & (PeakTable['Rot']=='P')].Wavenumber.count()
    PeakTable.loc[((PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37') & (PeakTable['Rot']=='P')),'Counter']=np.arange(-Psize,0)
    PeakTable.loc[((PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37') & (PeakTable['Rot']=='R')),'Counter']=np.arange(1,Rsize+1)
    
def btn_annotate(event):
    global PeakTable,fig,ax, annotationList,annotateFlag
    if annotateFlag==False:
        annotateFlag=True
    else:
        annotateFlag=False
    try:
        for annotation in annotationList:
            annotation.remove()
    except:
        annotationList=[]
    if annotateFlag:
        indexPeaks()
        for nuc1 in PeakTable['M1'].unique():
            yPeakMax=np.max(PeakTable[PeakTable['M1']==nuc1]['Absorbance'])
            yScaleMax=ax.get_ylim()[1]
            ytext=np.min([yPeakMax*1.2,yScaleMax/1.25])
            for nuc2 in PeakTable['M2'].unique():
                BooleanFilter=(PeakTable['M1']==nuc1) & (PeakTable['M2']==nuc2)
                if nuc2=='Cl37':
                    for index,x,y in zip(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],PeakTable[BooleanFilter]['Absorbance']):
                        newAnnotation=ax.annotate(str(index), xy=(x,y), xycoords='data', xytext=(x,ytext*1.1),  textcoords='data',arrowprops={'color': 'tab:green', 'width':1, 'headwidth':2},)
                        annotationList.append(newAnnotation)
                if nuc2=='Cl35':
                    for index,x,y in zip(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],PeakTable[BooleanFilter]['Absorbance']):
                        newAnnotation=ax.annotate(str(index), xy=(x,y), xycoords='data', xytext=(x,ytext*1.2),  textcoords='data',arrowprops={'color': 'tab:red', 'width':1, 'headwidth':2},)
                        annotationList.append(newAnnotation)
                    
def onpick(event):
    global PeakTable
    global fig,ax,h_cl37,h_cl35,d_cl35,d_cl37,h_low,d_low
    thisline = event.artist
    xdata = thisline.get_xdata()
    ind = event.ind
    wavenumber=xdata[ind][0]
    idx = PeakTable.index[PeakTable['Wavenumber']==wavenumber]
    index=idx[0]
    if PeakTable.at[index,'M2']=='Cl35':
        PeakTable.at[index,'M2']='Cl37'
    elif PeakTable.at[index,'M2']=='Cl37':
        PeakTable.at[index,'M2']='low'
    elif PeakTable.at[index,'M2']=='low':
        PeakTable.at[index,'M2']='Cl35'
    updateSpectra()

def saveToExcel(event):
    global file_path,data,PeakTable,FitTable
    indexPeaks()
    PeakTable=PeakTable.sort_values(by=['M2', 'M1', 'Counter'])
    
    writer = pd.ExcelWriter(file_path+"Fits.xlsx", engine='xlsxwriter')
    PeakTable.to_excel(writer, sheet_name='Peaks')
    workbook  = writer.book

    FitTable.to_excel(writer, sheet_name='Coefficients',startrow=1,startcol=0,header=False,index=True)
    worksheetCoef = writer.sheets['Coefficients']
    worksheetCoef.write('A1', 'Species')
    worksheetCoef.write('B1', 'Cx^3')
    worksheetCoef.write('C1', 'Cx^2')
    worksheetCoef.write('D1', 'Cx^1')
    worksheetCoef.write('E1', 'Cx^0')
    
    keyHClDCl=FitTable.reindex(['DCl37 coef','DCl35 coef','HCl37 coef','HCl35 coef'])
    keyHClDCl['M1']=np.array([mD,mD,mH,mH])/Na
    keyHClDCl['M2']=np.array([mCl37,mCl35,mCl37,mCl35])/Na
    keyHClDCl['u']=keyHClDCl['M1']*keyHClDCl['M2']/(keyHClDCl['M1']+keyHClDCl['M2'])
    keyHClDCl['vo']=keyHClDCl['Cx^0']
    keyHClDCl['a']=-keyHClDCl['Cx^2']
    keyHClDCl['D']=keyHClDCl['Cx^3']/-4
    keyHClDCl['B']=(keyHClDCl['Cx^1']+(2*keyHClDCl['a']))/2
    keyHClDCl['I']=h/(8*(np.pi**2)*c*keyHClDCl['B'])
    keyHClDCl['r']=np.sqrt(keyHClDCl['I']/keyHClDCl['u'])

#add formulas to calculate D, a, B, and vo based on the fit coefficients. Make sure these calculations have clear labels and units (wavenumbers for all four of these)
#wavenumber(as a function of index, m) = (-4D)m^3 + (-a)m^2 + (2B-2a)m^1 + (vo)m^0

#add formulas to calculate I (the moment of intertia in g*cm^2) and r (the internuclear separation in angstroms).
#I=h/(8*pi^2*c*B) (g*cm^2). I=u*r^2

#Comment the following line to avoid writing a sheet that does all calculations
    keyHClDCl.to_excel(writer, sheet_name='Key')

      
    for nuc1 in PeakTable['M1'].unique():
        for nuc2 in PeakTable['M2'].unique():
            if nuc2!='low':
                BooleanFilter=(PeakTable['M1']==nuc1) & (PeakTable['M2']==nuc2)
                sheetName="Fit_"+nuc1+nuc2
                worksheetFit = workbook.add_worksheet(sheetName)
                worksheetFit.write('B1', 'Index^3')
                worksheetFit.write('C1', 'Index^2')
                worksheetFit.write('D1', 'Index')
                worksheetFit.write('E1', 'Wavenumber (cm-1)')
                worksheetFit.write_column('B2',PeakTable[BooleanFilter]['Counter']**3)
                worksheetFit.write_column('C2',PeakTable[BooleanFilter]['Counter']**2)
                worksheetFit.write_column('D2',PeakTable[BooleanFilter]['Counter']**1)
                worksheetFit.write_column('E2',PeakTable[BooleanFilter]['Wavenumber'])
                numEntries=len(PeakTable[BooleanFilter]['Counter'])
                numIndex=str(numEntries+1)
                worksheetFit.write_array_formula('I3:L7', '{=LINEST(E2:E'+numIndex+',B2:D'+numIndex+',TRUE,TRUE)}')
                worksheetFit.write('I2', 'Cx^1')
                worksheetFit.write('J2', 'Cx^2')
                worksheetFit.write('K2', 'Cx^3')
                worksheetFit.write('L2', 'Cx^0')
                worksheetFit.write('H3', 'coefs')
                worksheetFit.write('H4', 'errors')
                worksheetFit.write('H5', 'r2, sy')
                worksheetFit.write('H6', 'F, df')
                worksheetFit.write('H7', 'SSreg, SSresid')
                chart1 = workbook.add_chart({'type': 'scatter'})
                chart1.add_series({
                    'name': 'Wavenumber',
                    'categories': [sheetName, 1, 3, 1+numEntries-1, 3],
                    #'values': '=Fits!$E$2:$E$28',
                    'values': [sheetName, 1, 4, 1+numEntries-1, 4],
                    'trendline': {
                        'type': 'polynomial',
                        'order': 3,
                        'display_equation': True,
                    },
                })
                chart1.set_title ({'name': nuc1+nuc2})
                chart1.set_x_axis({
                        'name': 'Index',
                        'min': PeakTable[BooleanFilter]['Counter'].min(),
                        'max': PeakTable[BooleanFilter]['Counter'].max()
                        })
                chart1.set_y_axis({
                        'name': 'Wavenumber (cm-1)',
                        'min': PeakTable[BooleanFilter]['Wavenumber'].min(),
                        'max': PeakTable[BooleanFilter]['Wavenumber'].max()
                        })
                chart1.set_style(6)
                worksheetFit.insert_chart('H8', chart1, {'x_offset': 25, 'y_offset': 10})
    workbook.close()
    writer.save()
    
def btn_plot(event):
    global PeakTable,FitTable
    indexPeaks()   
    fitCoefs={}
    fig2, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(16,9))

    BooleanFilter=(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35')
    ax1.plot(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],'^',color='tab:red')
    fit_DCl35=PolyReg(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],3)
    fitCoefs['DCl35 coef']=fit_DCl35['coefs']
    fitCoefs['DCl35 err']=fit_DCl35['errors']
    xRangeForFit=np.linspace(np.min(PeakTable[BooleanFilter]['Counter']),np.max(PeakTable[BooleanFilter]['Counter']),1000)
    ax1.plot(xRangeForFit,fit_DCl35['poly'](xRangeForFit),'-',color='tab:red')
    AnnotateFit(fit_DCl35,ax1,annotationText='Eq',xText=0.04,yText=0.94,boxColor='tab:red')
    AnnotateFit(fit_DCl35,ax1,annotationText='Box',xText=0.5,yText=0.2,boxColor='tab:red')
    
    BooleanFilter=(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35')
    ax2.plot(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],'o',color='tab:red')
    fit_HCl35=PolyReg(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],3)
    fitCoefs['HCl35 coef']=fit_HCl35['coefs']
    fitCoefs['HCl35 err']=fit_HCl35['errors']
    xRangeForFit=np.linspace(np.min(PeakTable[BooleanFilter]['Counter']),np.max(PeakTable[BooleanFilter]['Counter']),1000)
    ax2.plot(xRangeForFit,fit_HCl35['poly'](xRangeForFit),'-',color='tab:red')
    AnnotateFit(fit_HCl35,ax2,annotationText='Eq',xText=0.04,yText=0.94,boxColor='tab:red')
    AnnotateFit(fit_HCl35,ax2,annotationText='Box',xText=0.5,yText=0.2,boxColor='tab:red')
    
    BooleanFilter=(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37')
    ax3.plot(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],'^',color='tab:green')
    fit_DCl37=PolyReg(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],3)
    fitCoefs['DCl37 coef']=fit_DCl37['coefs']
    fitCoefs['DCl37 err']=fit_DCl37['errors']
    xRangeForFit=np.linspace(np.min(PeakTable[BooleanFilter]['Counter']),np.max(PeakTable[BooleanFilter]['Counter']),1000)
    ax3.plot(xRangeForFit,fit_DCl37['poly'](xRangeForFit),'-',color='tab:green')
    AnnotateFit(fit_DCl37,ax3,annotationText='Eq',xText=0.04,yText=0.94,boxColor='tab:green')
    AnnotateFit(fit_DCl37,ax3,annotationText='Box',xText=0.5,yText=0.2,boxColor='tab:green')
    
    BooleanFilter=(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37')
    ax4.plot(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],'o',color='tab:green')
    fit_HCl37=PolyReg(PeakTable[BooleanFilter]['Counter'],PeakTable[BooleanFilter]['Wavenumber'],3)
    fitCoefs['HCl37 coef']=fit_HCl37['coefs']
    fitCoefs['HCl37 err']=fit_HCl37['errors']
    xRangeForFit=np.linspace(np.min(PeakTable[BooleanFilter]['Counter']),np.max(PeakTable[BooleanFilter]['Counter']),1000)
    ax4.plot(xRangeForFit,fit_HCl37['poly'](xRangeForFit),'-',color='tab:green')
    AnnotateFit(fit_HCl37,ax4,annotationText='Eq',xText=0.04,yText=0.94,boxColor='tab:green')
    AnnotateFit(fit_HCl37,ax4,annotationText='Box',xText=0.5,yText=0.2,boxColor='tab:green')
    
    fig2.tight_layout()
    
    #FitTable=pd.DataFrame.from_dict(fitCoefs, orient='index',columns=['Cx^3', 'Cx^2', 'Cx^1', 'Cx^0'])
    FitTable=pd.DataFrame.from_dict(fitCoefs, orient='index')
    FitTable.columns=['Cx^3', 'Cx^2', 'Cx^1', 'Cx^0']                          

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
file_path=askopenfilename() #says that file path is the same as file name, opens up a dialog box to open a file
data=pd.read_csv(file_path,header=None,names=['wavenumber','absorbance'])
annotateFlag=False
annotationList=[]

lw=2550
hw=3120
V0=2889
peakHeightThreshold=0.01
Atom1='H'
SegmentY=np.array(data['absorbance'][(data['wavenumber']>=lw) & (data['wavenumber']<=hw)])
SegmentX=np.array(data['wavenumber'][(data['wavenumber']>=lw) & (data['wavenumber']<=hw)])
peakX,peakY,peakHeight=PeakFind(SegmentX,SegmentY)
peakHeightBool= (peakHeight>=peakHeightThreshold)

PeakSummary = pd.DataFrame(columns=['Wavenumber', 'Absorbance','M1', 'M2', 'Rot'])
PeakSummary['Wavenumber']=peakX
PeakSummary['Absorbance']=peakY
PeakSummary['M1']=Atom1
PeakSummary['M2']='low'
PeakSummary['Rot']='P'
PeakSummary.loc[PeakSummary['Wavenumber']>V0,'Rot']='R'
PeakSummary.loc[peakHeightBool, 'M2']='Cl37'
Threshold=PeakSummary[peakHeightBool]
ThresholdBool=Threshold['Absorbance'].diff()>0
idx=Threshold['Absorbance'][ThresholdBool].index
PeakSummary['M2'].loc[idx]='Cl35'

PeakTable=PeakSummary

lw=1870
hw=2260
V0=2090
peakHeightThreshold=0.01
Atom1='D'
SegmentY=np.array(data['absorbance'][(data['wavenumber']>=lw) & (data['wavenumber']<=hw)])
SegmentX=np.array(data['wavenumber'][(data['wavenumber']>=lw) & (data['wavenumber']<=hw)])
peakX,peakY,peakHeight=PeakFind(SegmentX,SegmentY)
peakHeightBool= (peakHeight>=peakHeightThreshold)

PeakSummary = pd.DataFrame(columns=['Wavenumber', 'Absorbance','M1', 'M2', 'Rot'])
PeakSummary['Wavenumber']=peakX
PeakSummary['Absorbance']=peakY
PeakSummary['M1']=Atom1
PeakSummary['M2']='low'
PeakSummary['Rot']='P'
PeakSummary.loc[PeakSummary['Wavenumber']>V0,'Rot']='R'
PeakSummary.loc[peakHeightBool, 'M2']='Cl37'
Threshold=PeakSummary[peakHeightBool]
ThresholdBool=Threshold['Absorbance'].diff()>0
idx=Threshold['Absorbance'][ThresholdBool].index
PeakSummary['M2'].loc[idx]='Cl35'
PeakTable=PeakTable.append(PeakSummary, ignore_index=True)

fig, ax = plt.subplots()
ax.plot(data['wavenumber'],data['absorbance'],'-k')
h_cl35,=ax.plot(PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35')]['Wavenumber'],PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl35')]['Absorbance'],'o', color='tab:red',picker=10)
h_cl37,=ax.plot(PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37')]['Wavenumber'],PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='Cl37')]['Absorbance'],'o',color='tab:green', picker=10)
d_cl35,=ax.plot(PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35')]['Wavenumber'],PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl35')]['Absorbance'],'^', color='tab:red', picker=10)
d_cl37,=ax.plot(PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37')]['Wavenumber'],PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='Cl37')]['Absorbance'],'^',color='tab:green', picker=10)
d_low,=ax.plot(PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='low')]['Wavenumber'],PeakTable[(PeakTable['M1']=='D') & (PeakTable['M2']=='low')]['Absorbance'],'^',color='lightgrey', picker=10)
h_low,=ax.plot(PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='low')]['Wavenumber'],PeakTable[(PeakTable['M1']=='H') & (PeakTable['M2']=='low')]['Absorbance'],'o',color='lightgrey', picker=10)
ax.invert_xaxis()
fig.canvas.mpl_connect('pick_event', onpick)

axcut = plt.axes([0.6, 0.0, 0.1, 0.05])
btn_ann = Button(axcut, 'Toggle Annotation', color='tab:gray', hovercolor='tab:blue')
btn_ann.on_clicked(btn_annotate)

axcut = plt.axes([0.7, 0.0, 0.1, 0.05])
btn_plt = Button(axcut, 'Plot', color='tab:gray', hovercolor='tab:blue')
btn_plt.on_clicked(btn_plot)

axcut = plt.axes([0.8, 0.0, 0.1, 0.05])
btn_save = Button(axcut, 'Save', color='tab:gray', hovercolor='tab:blue')
btn_save.on_clicked(saveToExcel)



